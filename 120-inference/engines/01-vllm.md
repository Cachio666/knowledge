---
title: vLLM 的架构选择：引擎循环、vllm.ir 的位置与版本相关的参数面
slug: inference-engines-vllm
domain: 120-inference
topic: engines
level: 3
difficulty: intermediate
prerequisites:
  - 120-inference/kv-cache/02-block-management
  - 120-inference/batching-and-scheduling/01-iteration-level-scheduling
tags: [vllm, engine-loop, max-num-seqs, chunked-prefill, prefix-caching, parameter-surface]
related:
  - 120-inference/engines/02-sglang-and-radix
  - 45-kernel-lifecycle/requirement-discovery/01-model-to-operator-inventory
  - 130-model-optimization/quantization/18-kv-cache-basics
code:
  - code/01_vllm.py
status: stable
updated: 2026-10-09
---

# vLLM 的架构选择：引擎循环、vllm.ir 的位置与版本相关的参数面

> **本文只写 vLLM 的【选择与差异】** —— 它的引擎循环长什么样、`vllm.ir` 在它的架构里占哪个位置、
> 以及一件常被写错的事：**它的几个调度参数【没有单一默认值】**。
>
> ★★★ **本主题的硬约束**：**「引擎的机制」一律回链** ——
> **分页 / 块管理** → [02-block-management](../kv-cache/02-block-management.md)；
> **前缀共享** → [03-prefix-and-eviction](../kv-cache/03-prefix-and-eviction.md)；
> **请求调度器** → [01-iteration-level-scheduling](../batching-and-scheduling/01-iteration-level-scheduling.md)；
> **PagedAttention 的定义与算式** → `130-model-optimization/quantization/18-kv-cache-basics:68-84` +
> `40-kernels/attention/05:130-179`。
> ★★★ **本主题【没有 A 类】** —— 本机**无 GPU、无任何引擎**（vLLM / SGLang / TensorRT-LLM / llama.cpp 都没有）
> ⇒ **vLLM 的【行为与性能】无处可验**。★★ 「不是任何真实引擎，也不是真机」。
> ★ 但**它的数据结构仍可验** —— 那是 [02-sglang-and-radix](02-sglang-and-radix.md)（本主题**唯一**有 A 类的一篇）。

## 0. 背景与动机

一个推理引擎要做的事，可以压成三句：**把请求排进队、把算力喂满、把显存管好**。
「怎么管显存」的正文在别处（见上面的回链）；**本文要讲的是前面那件事的【形状】**：

**引擎的主循环长什么样？它每一轮（一次 iteration）做什么决定？**

★ 为什么这件事值得单列：**它是「引擎」区别于「一个前向函数」的地方**。
一个前向函数只回答「给定这一批输入，输出是什么」；引擎要回答的是
「**这一轮该挑哪些请求、放多少 token 进去**」—— 而**这个决定随版本、随设备、随场景变**。

★★★ **一个具体的对照场景**（本文程序里的真算例，`code/01_vllm.py` 的 `B.1`）：
你最可能在别处读到的一句话是「vLLM 的 `max_num_seqs` 默认是 `128`」。
★ 本文程序把 vLLM 源码里那张 **设备 × 场景** 的表摆出来之后，**这句话的直接后果**是：

| 你若照抄 `128` | 实际会发生什么 |
|---|---|
| **A100 上跑 API server** | 实际默认是 **`256`** —— ★ 你的「默认值」少了一半 |
| **B200/B300 上跑 API server** | 实际默认是 **`1024`** —— ★ 差了 **8 倍** |
| **开了 `throughput` 模式** | 上面每个数**再翻一倍** |

⇒ ★★★ **「vLLM 的 `max_num_seqs` 默认值」这句话【没有单一答案】** ——
它取决于**设备显存档**与**使用场景**。把这句写成一个数字，就是**「规范与实现不分」**的现场。

## 1. 原理

### 第一步：引擎循环 —— vLLM 自己的措辞

vLLM 官方文档 `design/arch_overview` 逐字把它写成（`[B]` 级，vLLM 官方文档）：

> 「a busy loop that continuously schedules requests and dispatches work to the GPU workers」

★★★ **两点必须说清**（否则会被读成别的东西）：

1. **它是「一个忙循环」** —— 每一轮（一次 iteration）做一次**调度决定**，再把结果**派发给 GPU worker**。
   ★ **这就是「引擎」与「前向函数」的分界**。
2. ★★ **该页【没有】使用「continuous batching」这个词** —— 首页只把它当**特性名**列出
   （逐字「Continuous batching of incoming requests, chunked prefill, prefix caching」，`[B]`）。
   ⇒ ★★★ **不得写「vLLM 定义 continuous batching 为……」** ——
   本批取数**找不到**该术语在 vLLM 文档里的**逐字机制定义**。
   ★ 那个**原始术语**（`iteration-level scheduling`）的出处是 **ORCA（OSDI '22，`[C]`）**（见 §2 与 §4.3）。

### 第二步：`vllm.ir` 在架构里的位置 —— 一处**回链**

vLLM 把「**算子语义**」与「**算子实现**」分成两层，**中间那层叫 `vllm.ir`**。
★ **本文只记【它的名字与位置】** —— ★★★ **那套注册机制的四项细节（`@register_op` / `register_impl` /
优先级列表 / `native` 兜底）本文一处都不重写**，逐项回链见下。

★★★ **这套机制的正文【不在本文】** —— 它在
`45-kernel-lifecycle/requirement-discovery/01:205-214`。★ 本文只记**它的位置**：

**它是引擎循环的【下游】。** 循环决定「这一轮处理哪些请求」，而**每个请求要跑哪些算子、
每个算子用哪份实现**，由 `vllm.ir` 的注册表回答。
⇒ ★ **换一份实现（比如把某个算子切到 Triton 版本）不需要改循环** —— 这是这一层分开的全部意义。

★ **本文一行都不重写它**（`P37`：一个知识点只有一处正文）。

### 第三步：★★★ 参数面 —— 为什么「默认值」不是一个数

vLLM 的调度参数里，最容易写错的是这两个：`max_num_seqs` 与 `max_num_batched_tokens`。
文档对二者的说明**逐字**都带同一句（`[B]` 级）：

> 「The default value here is **mainly for convenience when testing**. In real usage,
> this should be set in `EngineArgs.create_engine_config`.」

★★ **文档自己说这个 default 不是生产默认**。真正的默认由 `get_batch_defaults()` 按
**设备显存档 × 使用场景**覆盖（源码，`main` 分支）：

| 设备档 | 场景 | `max_num_batched_tokens` | `max_num_seqs` | 每槽 token 数 |
|---|---|---|---|---|
| **B200/B300** | API server | `16384` | `1024` | `16` |
| **B200/B300** | 离线（`LLM_CLASS`） | `16384` | `1024` | `16` |
| **H100/H200** | API server | `8192` | `1024` | `8` |
| **H100/H200** | 离线（`LLM_CLASS`） | `16384` | `1024` | `16` |
| **A100 / 其余** | API server | `2048` | `256` | `8` |
| **A100 / 其余** | 离线（`LLM_CLASS`） | `8192` | `256` | `32` |

★★★ **从这张表能读出三件事**（本文程序的 `B.1` / `B.2` 把它们算成可断言的量）：

1. ★★ **「兜底」与「真实默认」不是一回事**：类常量 `DEFAULT_MAX_NUM_SEQS = 128` /
   `DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048` 只是**兜底**；表里的值才是**覆盖后的**。
2. ★★ **加上 `throughput` 模式（未显式给值时两个参数【各自翻倍】）**，`max_num_seqs` 的
   **取值集合**是 `{128, 256, 512, 1024, 2048}` —— **5 个不同取值**。
3. ★★★ **两个参数【不是等比缩放的一对】**：**每槽 token 数**（`max_num_batched_tokens / max_num_seqs`）
   在表里取 `8 / 16 / 32`（**跨 4 倍**）。★ 若它们是等比的一对，这个比值应当**恒定**。
   ★ 注意 `throughput` 模式**两个都翻倍** ⇒ **比值不变** —— 这也说明翻倍是「整机扩容」而不是「换一种配比」。

★ **一条读法**：**参数对是被两个独立的预算各自定的**，不是「一个数乘以固定的比例」。

★★★ **★★ 这张表的【唯一正文在此】**（`CONTRIBUTING` §7，**收拢已完成**）—— 它是 **vLLM 专有的配置面**，
去掉引擎层即消失 ⇒ 归**引擎层**。★ 本域的
[01-iteration-level-scheduling](../batching-and-scheduling/01-iteration-level-scheduling.md)
**已收成一行的结论 + 回链**（★ 那里的角度是它对**请求调度**的影响，不是引擎配置面）。
★ 本文的角度是**「引擎配置面」**（含 `throughput` 翻倍与**每槽 token 数**）。

## 2. 执行映射

引擎层。本文的每个「量」都是**引擎层的配置面，不是任何硬件数字**。

**（1）引擎循环里的一次「步」（step）**：`design/arch_overview` 的措辞是
「a busy loop that continuously schedules requests and dispatches work to the GPU workers」。
★ 每一步的**形状**是：「挑请求 → 组批 → 派发给 GPU worker」。
★★ 但**每一步具体挑谁、放多少 token 进去**，就是下面那组参数在决定的。

**（2）★★★ 参数面就是引擎的「档位表」**：上表 6 行里，`max_num_seqs` 只在两个值上（`256` / `1024`），
而 `max_num_batched_tokens` 在三个值上（`2048` / `8192` / `16384`）。
★ **两者的组合不是自由的** —— 它由「设备显存档 × 场景」唯一确定。

**（3）★ 一个必须连限定一起写的默认值 —— chunked prefill**：
文档逐字（`[B]`）「**In V1**, chunked prefill is enabled by default **whenever possible**」。
★★★ **两个限定缺一不可**：
- **`In V1`**：这是 **V1 引擎**的语义（V0 的调度策略不同）；
- **`whenever possible`**：**encoder-decoder 模型会禁用它**（`SchedulerConfig` 的 docstring 明说）。

⇒ ★★ **写「chunked prefill 默认开启」而不带这两个限定，就是在别的语义下写错了。**

**（4）★ 另一个「默认值未取到」的开关 —— prefix caching（APC）**：
文档只写了开关本身（逐字「`--enable-prefix-caching` … Whether to enable prefix caching.」），
★★ **取数到的页面【未声明】它的开/关默认值** ⇒ **不得断言「APC 默认开启」**。
★ 对照：**SGLang 的 RadixAttention 是无开关的默认行为** —— 这一条在
[02-sglang-and-radix](02-sglang-and-radix.md) 的 `C.2` 里**并排写出**。

**（5）★★ 这条链的产物是「配置面」，不是「真实性能」。**
本机**无 GPU、无 vLLM** ⇒ 上面的数**没有一个**来自真实系统。
★★★ **限定语**：「不是任何真实引擎，也不是真机」。

## 3. 实现与演进

**第一步：一个前向函数。** 只回答「给定输入，输出是什么」。**没有循环、没有调度** ——
★ 失败模式：**请求之间互相等待**（一个长请求拖住整批），这是**静态批**的形态。

**第二步：引擎循环 + 迭代级调度。** 把「调度」变成一个**每一步都重做**的决定。
★ 术语出处：`iteration-level scheduling`，**ORCA（OSDI '22）**。
★★ **一处必须写清的归属**：`continuous batching` 这个叫法是**产品文档**里流通的
（NVIDIA 对 TensorRT-LLM 自己写「in-flight batching … also known as continuous batching or
iteration-level batching」，`[A]`级，见 [03-tensorrt-llm-and-llamacpp](03-tensorrt-llm-and-llamacpp.md)）。
⇒ ★★★ **不得写「continuous batching 由 vLLM 提出」**。

**第三步：把循环的参数化。** 一旦「每步挑谁」变成配置，就能按**设备/场景**给不同档位 ——
★ 这就是 §1 那张表的由来。★★ **代价是**：同一个参数在**不同档位上不是同一个数**，
⇒ **写文档时「一个数」这种做法不再成立**（`F2`）。

**第四步：把算子层与循环层分开（通往 `vllm.ir`）。** 循环只管「挑请求」，
「跑哪些算子、用哪份实现」交给 **`vllm.ir`** 的注册表（★ 那四项注册细节**正文在彼处**，本文**不重列**）。
★ 这一步让**换实现不动循环**成为可能 —— ★ 机制正文在 `45-kernel-lifecycle/requirement-discovery/01:205-214`。

## 4. 踩坑与失败模式

### 4.1 把「兜底常量」当成「vLLM 的默认值」

- **现象**：文档里写「vLLM 的 `max_num_seqs` 默认是 `128`」，读起来像个确定的事实；
- **成因**：`128` 是**类常量兜底**；真实默认由 `get_batch_defaults()` 按**设备 + 场景**覆盖；
  ★ **A100 上 API server 的真实值是 `256`**；
- **本库判据**（`code/01_vllm.py` 的 `C.1`）：**不带设备与场景**的表述被**拒**（反例 → 被拒）；
  取值集合是 **5 个**（`128/256/512/1024/2048`）⇒「单一默认值」这个论断**判为不成立**；
- **规避**：★★★ **写这个参数时【必须带设备与场景】**；★ 引不出两个条件，就**别写默认值**。

### 4.2 「默认开启」丢了限定

- **现象**：写「chunked prefill 默认开启」，读者于是在 V0 语义 / encoder-decoder 模型上也这么以为；
- **成因**：★★ 原句是「**In V1**, … enabled by default **whenever possible**」——
  **两个限定都是句子的一部分**，不是修饰；
- **本文实测**（`code/01_vllm.py` 的 `C.2`）：完整表述（带两个限定）**通过**；
  裸表述「chunked prefill is enabled by default」**被拒**；
- **规避**：★★ **把限定词当成句子的必要成分**；★ 去掉它，这句话在别的语义下就是错的。

### 4.3 把 `iteration-level scheduling` 当成 vLLM 造的术语

- **现象**：写「vLLM 提出了 iteration-level scheduling」；
- **成因**：★ 该术语的**原始出处是 ORCA（OSDI '22）**；vLLM 首页只把 `continuous batching`
  当**特性名**列出，**没有机制定义**；
- **本库判据**（`code/01_vllm.py` 的 `C.3`）：把该术语**归给 vLLM** 会**被拒**（反例 → 被拒）；
  归给 **ORCA（OSDI '22）**才通过；
- **规避**：★ **引术语必引出处**；★★ 写「continuous batching」时，同句说明它是**产品文档的流通叫法**。

### 4.4 裸写一个社区缩写，当成官方的词

- **现象**：写「vLLM 用 `TPOT` 衡量每 token 延迟」；
- **成因**：★★ vLLM 的官方词是 **inter-token latency**（`mean_itl_ms` /
  `request_time_per_output_token_seconds`）；**该缩写在 vLLM 的两个 metrics 页面上出现 0 次**；
- **本库判据**（`code/01_vllm.py` 的 `C.4`）：把它当**官方词**会**被拒**（反例 → 被拒）；
  该缩写计数 = **0**；
- **规避**：★★ **写社区缩写时注明「这是社区常用词」**，并给出 vLLM 的官方词。

### 4.5 引一个「被官方自标为历史文档」的页，当成现状

- **现象**：拿 vLLM 的 `design/paged_attention` 页当 vLLM 当前实现的正典；
- **成因**：★★★ 该页**自己**写着「This is a historical document based on the original paper for
  vLLM. **It no longer describes the code used in vLLM today.**」；
- **本文实测**（`code/01_vllm.py` 的 `D.6`）：该句与论文定义**逐字钉住**，
  并由 `C.1`/`F7` 一类判据拦住误用；
- **规避**：★★★ **引任何工件前先读它自己的「不做什么 / 已作废什么」** ——
  它比「做什么」那一段更能定案（`CONTRIBUTING` §2 的同一手法）。

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| 见到「vLLM 的 `max_num_seqs` 默认是 `N`」 | ★★★ **必须带设备与场景** —— 取值集合是 `{128,256,512,1024,2048}`（兜底 + 6 格 + throughput 翻倍）⇒ **没有单一默认值**（`B.1`/`C.1`，`F2`） |
| 见到「vLLM 定义 continuous batching 为……」 | ★★★ **找不到逐字支撑** —— 首页只当**特性名**列出；机制定义要引 **ORCA 的 `iteration-level scheduling`**（`C.3`，`F3`） |
| 见到「continuous batching 由 vLLM 提出」 | ★★ **错** —— 原始术语出处是 **ORCA（OSDI '22）**（`C.3`） |
| 见到「chunked prefill 默认开启」 | ★★ **必须带 `In V1` + `whenever possible`** —— 缺任一个即不完整（`C.2`，`F6`） |
| 见到「APC 默认开启」 | ★★★ **不得写** —— 取数到的页面**未声明**开/关默认值（`F5`） |
| 见到「`TPOT` 是 vLLM 的官方词」 | ★★ **错** —— 该缩写计数 **0**；官方词是 **inter-token latency**（`C.4`，`F9`） |
| 见到「引 `design/paged_attention` 描述 vLLM 现状」 | ★★★ **不得** —— 该页**自标为历史文档**（`D.6`，`F7`） |
| 见到 `36:9×` 或 `23x` | ★★★ **是【引用值】**（ORCA 论文 / 一篇博客，两个不同来源）；基线各异，**不得当作本库的测量**（`D.5`，`F4`） |
| 见到「引擎循环」 | ★ 用 vLLM 的措辞：**a busy loop that continuously schedules requests and dispatches work to the GPU workers**（`B.1`） |
| 见到 `vllm.ir` / `@register_op` / `register_impl` / 优先级列表 | ★★★ **不在本文** —— 正文在 `45-kernel-lifecycle/requirement-discovery/01:205-214`，此处回链 |
| 见到「PagedAttention 的算式 / 分页机制」 | ★★★ **不在本文** —— `130-model-optimization/quantization/18-kv-cache-basics:68-84` + `40-kernels/attention/05:130-179` + 本域 `kv-cache/02` |
| 见到「vLLM 的真实吞吐 / 延迟」 | ★★★ **本机测不了**（无 GPU、无 vLLM）⇒ 只能引 `[B]`/`[C]` 级值，**必须带条件** |

## 6. 小结与延伸

**一句话**：**vLLM 的引擎循环是一个「持续调度」的忙循环（`design/arch_overview` 的措辞），
而它的调度参数是【按设备与场景给的一张表】—— 把那张表压成一个数字，就丢掉了它的语义。**

**三条可独立复用的结论**：

1. ★★★ **「默认值」不是标量、是函数**：见到「引擎的某参数默认是 N」，先问**在哪个设备、哪个场景**；
2. ★★ **限定词是句子的一部分**：`In V1` / `whenever possible` 去掉一个，语义就变；
3. ★★★ **引工件前先看它的自述**：一个被官方标为「历史文档」的页，**不能当现状**。

**延伸**：

- ★★ **SGLang 的 RadixAttention**（前缀树 / LRU / 两处口径差）→ [02-sglang-and-radix](02-sglang-and-radix.md)；
- ★★ **TensorRT-LLM 的编译式路线**与 **llama.cpp 的边界** → [03-tensorrt-llm-and-llamacpp](03-tensorrt-llm-and-llamacpp.md)；
- ★★★ **vLLM 的模型接入与算子注册（`vllm.ir`）** → `45-kernel-lifecycle/requirement-discovery/01:205-214`（**正文在彼处**）；
- ★★★ **PagedAttention 的定义与算式** → `130-model-optimization/quantization/18-kv-cache-basics:68-84`（**正文在彼处**）；
- ★★ **块管理 / 分页的系统侧机制** → 本域 [02-block-management](../kv-cache/02-block-management.md)；
- ★★ **前缀共享与淘汰（`hash → 块`）** → 本域 [03-prefix-and-eviction](../kv-cache/03-prefix-and-eviction.md)；
- ★ **请求调度器的机制** → 本域 [01-iteration-level-scheduling](../batching-and-scheduling/01-iteration-level-scheduling.md)。

## 参考

- **`[B]` vLLM 官方文档**（`design/arch_overview` · `configuration/engine_args/` ·
  `configuration/optimization/` · `features/per_request_metrics/` · 首页）——
  文档站 = **latest developer preview**（页脚 2026-10-08），对应发布版 **0.31.0**；抓取日 **2026-10-09**。
- **`[B]` vLLM 源码**（`main` 分支，抓取日 2026-10-09）—— `vllm/config/scheduler.py`
  （`DEFAULT_MAX_NUM_SEQS` / `DEFAULT_MAX_NUM_BATCHED_TOKENS`）· `vllm/engine/arg_utils.py`
  （`get_batch_defaults`）。★ **行号会漂移 ⇒ 锚点用「文件路径 + 常量名」**。
- **`[C]` ORCA（OSDI '22）** —— `https://www.usenix.org/conference/osdi22/presentation/yu`；
  原始术语 `iteration-level scheduling`；结果值 `36:9×`（★ **引用值**，基线 = NVIDIA FasterTransformer）。
- **`[C]` PagedAttention 论文** —— arXiv:2309.06180（SOSP 2023）。
- ★ **一条术语纪律**：本文一律写「**KV 缓存**」「**请求调度器**」全称；
  ★ `block` / `page` 是**同名不同指**（见 [02](02-sglang-and-radix.md) 的 `C.1`）。
- ★ **跨域回链**（**正文在彼处，本文只回链**）：
  · `vllm.ir` 与模型接入 → `45-kernel-lifecycle/requirement-discovery/01:205-214`；
  · PagedAttention 的算式 → `130-model-optimization/quantization/18-kv-cache-basics:68-84`；
  · 分页的 kernel 侧 → `40-kernels/attention/05:130-179`。

## 编译与运行

```sh
cd code && make run                       # 三个编号程序，任一断言失败即非零退出
python3 01_vllm.py                        # 只跑本文的配套程序
make 01_vllm                              # ★★ 与上一行**不等价**：它先过 check-env（查解释器 / numpy）再跑
```

★ 本机的运行条件：`python3 3.13.5`、`numpy 2.4.4`、macOS arm64，**无 GPU、无任何引擎**。
★★★ **本文的程序【没有 A 类】**（本机无引擎 ⇒ 引擎的行为与性能无处可验）——
`D.0` 用 **AST 扫自己的源码**证明 `check("A…")` 的调用数 == `0`，**一切钉住归 `D` 类**。
★ `B.1`/`B.2` 是**参数面的结构账**；`C.1`–`C.4` 是四条**禁写的真现场**（`F2`/`F6`/`F3`/`F9`）；
★ `D.9` 是**术语纪律**（`裸写被禁裸词的计数 == 显式豁免清单覆盖数`）。
★ **本文对应 `code/01_vllm.py`**（三类断言全在程序内部，失败即非零退出）。

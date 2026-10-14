# 120-inference 推理引擎与部署

从权重到服务。训练追求吞吐，推理要同时面对**吞吐、延迟与显存**三重约束，而这三者互相交换。

## 本域覆盖什么

| 主题 | 内容 |
|---|---|
| kv-cache | ★ **从「账」到「容量规划」**（可用显存 → KV 块数 → 最大并发）· **系统侧的块管理器**（块池 / 引用计数 / copy-on-write）· **前缀共享与淘汰**（hash → 块、驱逐、命中率） |
| batching-and-scheduling | **迭代级调度**（continuous batching 的机制与收益来源）· **chunked prefill 的分块数权衡** · **抢占与公平** |
| decoding-strategies | 采样与搜索（greedy / beam / top-k / top-p）· ★★ **speculative decoding**（三条独立路径 + 假设边界）· 并行解码 |
| engines | vLLM · SGLang（RadixAttention）· TensorRT-LLM · llama.cpp —— ★★ **只写「各家的选择与差异」，机制一律回链**。★★ **`llama.cpp`：本次取数无一手来源 ⇒ 只作【边界声明】**（`03` 的 §0.5 / `C.3`，`F3`） |
| serving-infra | **两个延迟**（TTFT / inter-token latency）· ★★ **推理的 TP/PP 与 PD 分离**（★★ **EP 显式不在本域** —— 见 `serving-infra/02` §1.4）· 扩缩容 / 路由 / 可观测性 |

## 主线

> »**训练是「一个批走到底」，推理是「请求不断进出」—— 这个差别让推理多出整整一层：调度。
> 而调度的每个决策都在【显存 / 吞吐 / 延迟】三者之间做交换；三个账都能算，
> 但算出来的是【机制与上界】，不是【真实系统的性能】。**«

| 约束 | 它的账 | 本机可验性 | 正文位置 |
|---|---|---|---|
| **显存** | KV cache 是**动态增长**的 ⇒ 必须**按需分配** | ★ **可算**（容量 / 块数 / 并发） | ★ 本域 `kv-cache`；★★ **「账」本身在 `130-model-optimization/quantization/18-kv-cache-basics`，本域只回链** |
| **吞吐** | 批越大吞吐越高 ⇒ 但受显存与延迟限制 | ★ **可模拟**（调度器） | ★ 本域 `batching-and-scheduling` |
| **延迟** | ★ **有两个**：TTFT 与 inter-token latency；★★ **它们【何时】冲突取决于模型** —— ★ 见 `serving-infra/01` 的 §3（★ 本文模型里 `B` 越大 TTFT 反而越小，因为 prefill 记成独立一步） | ▲ 可推导 | ★ 本域 `serving-infra/01` |
| **实现** | 各家引擎怎么选（分页 / 前缀树 / 调度器） | ✗ **不可验**（本机无 GPU、无引擎） | ★ 本域 `engines`（**只有 `[B]` 级**） |

★★★ **推论（本域最要紧的一条）**：**「账能算」推不出「真实系统会这样跑」** ——
★★ 本机**无 GPU、无 vLLM / SGLang / TensorRT-LLM**（实测，见计划 `§0.7 A`）
⇒ **任何真实引擎的数字都是引用值**，必须带**级别 + 版本 + 条件**。
★ **每篇文档都在抬头写明它写的是哪一层。**

★★ **本域的背景**：它是**全库最后一个建成的域**，因此它的**大部分知识对象的正文已经在别处**（见下表）。
⇒ ★★★ **本域的头号纪律是「先查库，再动笔」**：任何一篇开写前，必须先在库内搜一遍它的知识对象有没有正文。

## 与其他域的关系

- **上游**：`80-model-architectures`（KV cache 的大小由架构决定）· `40-kernels`（引擎的算力上限由 kernel 决定）· `60-compilers`（引擎依赖编译期优化）· `70-frameworks` · `130-model-optimization`（量化是推理加速的主要手段）。
- **下游**：无。这是栈的最上层。
- **横切**：KV cache 量化（正文在 `130-model-optimization/quantization`，此处回链）。

★★ **反向的契约（本域要主动去补的）**：`45-kernel-lifecycle` **全域 0 处**提到本域，
★ 但它的 `requirement-discovery/01:158-214` 写了 **vLLM 的模型接入三档、算子注册、缓存层重构** ——
★★ 本域建成后**主动**在此处补回链（★ 视角不同，**不构成重复建设**）。

## 本域不做什么（硬边界）

下表是**硬边界**：本域的主题**不写**「本域不写」那一列的内容，只在正文里**回链**。
回链的注记方向按 `CONTRIBUTING` §7（「正文在 X，此处回链」与「正文在本域，回链 Y」两种，**不可混用**）。

| 主题 | 本域**不写** | 那里的正文 |
|---|---|---|
| **全主题** | ★★★ **KV cache 的「账」**（六项乘积、`512 KiB` / `2 GiB`、GQA 的坑、`AI = 2·S·T/(bpe·(S+T))`）与 **「预留 vs 分页」的浪费对比** | `130-model-optimization/quantization/18-kv-cache-basics:68-84,124` —— ★★ 本域**只写「从账到容量规划」与「系统侧的块管理」** |
| **全主题** | ★★★ **paged KV 的 kernel 侧【全段】**：页表延迟（400–800 周期 vs tile 计算 504 周期）· 碎片（`16/2 → 0.049%`）· 布局必须「一行」连续 · **块大小的双向权衡及其结论**（「真正约束是查表次数，不是碎片」） | `40-kernels/attention/05:130-179,283-290` —— ★★ **全部只回链** |
| **全主题** | ★ **decode 的 attention kernel**（GQA/MQA 按 `1/r` 砍字节 · split-KV · decode 的 `AI = r`） | `40-kernels/attention/05`（全篇）+ `10-foundations/performance-models/01:147-155` |
| **全主题** | ★★ **KV cache 量化**（K/V 的轴、低比特、各家引擎把 scale 放哪一维） | `130-model-optimization/quantization/19-kivi-asymmetry,03` |
| **全主题** | ★ **vLLM 的模型接入流程与算子注册**（三档分治 · `vllm.ir` · `@register_op` · KDA 的 prefill/decode 拆分） | `45-kernel-lifecycle/requirement-discovery/01:158-214`（+ `operator-design/01` / `implementation/01`） |
| **全主题** | ★ **continuous batching / chunked prefill 的【定义】** | `10-foundations/performance-models/01:251` —— ★ 本域只写**它们的机制与收益来源** |
| **`batching-and-scheduling`** | ★ **prefill/decode 的【kernel】差异**（两者的 `AI` 差 120 倍） | `40-kernels/attention/05:229-236` |
| **`serving-infra`** | ★ **通信的机制**（集合通信原语、NCCL、拓扑、重叠） | ★ `110-communication` 全域 —— 本域只写**推理侧的通信需求**（KV 传输、TP/PP 的形态） |
| **`serving-infra`** | ★ **EP（专家并行）的推理侧** | ★ `100-parallelism/expert-parallel` + `110-communication/collectives` —— ★★ **本域显式声明**它「**与训练侧无实质差异**」（`serving-infra/02` §1.4），★ 不是默认省略 |
| **`serving-infra`** | ★ **并行策略本身的正文**（数据/张量/流水/序列/专家/状态六轴） | ★ `100-parallelism` 全域 —— 本域只写**推理与训练的实质差异**（PP 无反向 ⇒ 气泡模型不同；TP 要切 KV cache） |
| **`engines`** | ★★ **引擎的机制细节**（分页 → `kv-cache/02`；前缀树 → `kv-cache/03`；调度器 → `batching/01`） | ★ 本域其它篇 —— `engines` **只写「各家的选择与差异」** |
| **全主题** | ★ **训推共用的导出格式与权重账** | `70-frameworks/tensorflow/03` + `90-training/checkpointing/01` |
| **`decoding-strategies`** | ★ **框架侧的采样 API**（`temperature` / `top-p` 在某个框架里的**参数名与默认值**） | ★★ **本库无此正文**（`70-frameworks` 的三个主题是 dispatch / autograd / 内存，**不涉采样**）—— ★ 这是一条**边界声明**，**不是**「正文在别处」 |

## 同名不同指（六个 · ★ 每个都已核过）

| 词 | 库内已有的含义 | 本域的处置 |
|---|---|---|
| ★★ **`page` / `block`** | ★★★ **两个实现的粒度不同**：**vLLM 的 block = 16 token** vs **SGLang 的 page = 1 token**（差 16 倍） | ★★ 引用时必须**写明是哪一家** |
| **`填充浪费`** | ★★ **`1 − 均值/最大值`**（`40-kernels/moe-kernels/02:89`）—— 那是**一个 batch 内的槽位/算力浪费** | ★★ **同式不同指**：本域的**同一式**描述的是「静态批等最长」**在时间上的浪费**；★ 首次出现处点明两者是同一个式子 |
| **`重算`** | ★★ **已有三义**：编译器的「物化 vs 重算」· 训练的「重算换显存」（gradient checkpointing）· ★ 本域的「**KV 重算**」（抢占时丢 KV、需要时重算） | ★ 本域一律写「**KV 重算**」全称 |
| **`batch`** | 训练的 mini-batch · `40-kernels` 的 tile | ★ 本域写「**推理批 / 在飞请求数**」 |
| **`scheduler`** | GPU 的 warp scheduler · `60-compilers` 的调度器 | ★ 本域写「**请求调度器**」 |
| **`decode`** | 数值的 decode（`60-compilers` 的 decode pass） | ★ 本域写「**自回归解码**」 |

## 代码约定

- **编号程序**：`.py`，**默认 `python3` 可跑**；★ **不 import 任何第三方框架**
  （实测：默认 `python3` **没有 `torch`**，见计划 `§0.7 A`），`numpy` 与标准库为限。
- ★ **单进程** · ★★ **本域不设 `probes/`**（★ 没有「必须真跑多进程」的知识对象，与 `110-communication` 不同）。
- **三类断言（A / B / C）缺一不可**，全部**在程序内部**，失败以非零退出码结束。**不设 `verify.py`**。
- ★ **§5 采用「判据表」形态**（`CONTRIBUTING:380`）：本域**无 GPU、无法真跑目标系统**，且知识对象是**方法**。
- ★★★ **本域最危险的判据是 `P37`（把库内已有的推导再算一遍）** ——
  ★ **开写前先 `grep` 核心符号**；一个公式或数字在库里出现两次，就是 `P37`。
- ★★ **`engines` 里「真实引擎的行为与性能」没有 A 类**（本机无引擎）——
  ★ 但**它的数据结构（前缀树 / 块池 / LRU）有真 A 类**。
- ★ **限定语（本域最强的）**：**「不是任何真实引擎，也不是真机」** —— 本域的一切数字都是
  **推导值 / 引用值**；★ **引用外部数字必须带级别（A–E）+ 版本 + 条件**。

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)

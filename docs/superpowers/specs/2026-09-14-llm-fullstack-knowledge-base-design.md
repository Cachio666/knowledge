# 大模型全栈知识库 设计方案

- **日期**：2026-09-14
- **状态**：待实施（第 2 轮，已按审查意见修订）
- **目标仓库**：`/Users/cachio/gitProjects/CUDA/knowledge`

---

## 1. 背景与目标

### 1.1 背景

需要一个用于学习的技术知识库。学习目标是**大模型从硬件底层到引擎上层的完整链路**，涵盖训练与推理的全部内容。最初的切入点是 CUDA 算子开发、模型量化、GPU 编程、PyTorch、TensorFlow 五项，但经确认这五项只是入口，知识库的边界应是**整个大模型技术栈**。

### 1.2 当前仓库状态

- `knowledge/` 已初始化为 git 仓库，已有首次提交（本设计文档）。
- 目录内容仅为本设计文档与 `.gitignore`，知识内容为空。
- 同级参考项目（**不依赖、不迁移**）：
  - `/Users/cachio/gitProjects/CUDA/` 下的 `cuda-learning`、`cuda-training-series`、`cutlass`
  - `/Users/cachio/gitProjects/` 下的 `CUDA_Freshman`、`Pytorch-Inductor-Tutorial`

### 1.3 执行环境前提

| 环节 | 环境 |
|---|---|
| 文档编写、仓库管理 | 本机 macOS（**无 CUDA 工具链，无 `nvcc`**） |
| `.cu` 代码编译与运行 | 远程 Linux 机器，CUDA Toolkit ≥ 12.0，GPU 算力 ≥ sm_80 |

**含义**：知识库中的 CUDA 代码**不在本机编译**。验收时须将 `code/` 目录同步到远程机器执行。文档中涉及实测数字时，须注明测量所用 GPU 型号与 CUDA 版本；若未实测，须明确标注为**推导值**。

### 1.4 目标

1. 建立覆盖大模型全栈的**可扩展**目录体系，新领域可直接插入而不触发现有结构。
2. 文档**深入原理**，而非资料摘抄。深度由可检查的维度标准定义（见 7.2）。
3. 文档与代码**明确区分**，且两者保持可验证的对应关系（见 5.1）。
4. 全部采用 Markdown 格式。
5. 同时服务于**人阅读**（连贯、可导航）与**机器检索**（结构化元数据、语义边界清晰）。
6. 维护一份**简单明了、易于维护的全库进度表**（见 9）。

### 1.5 非目标（YAGNI）

- **不做静态站点生成**（VitePress / MkDocs 等）。当前阶段纯 Markdown + 编辑器/文件系统导航即可。
- **不做向量化入库与检索服务**。frontmatter 与文件切分规则为将来留好接口，但现在不实现。
- **不迁移或拆分现有同级项目**。知识库自包含、不依赖它们。
- **不写索引生成脚本**。索引与进度表首期手工维护（见 10 的触发条件）。
- **不追求内容全覆盖**。首期只交付骨架与一条完整的样板链，其余目录放一级域 README 占位。

---

## 2. 需求决策记录

| 编号 | 决策点 | 结论 | 影响章节 |
|---|---|---|---|
| D1 | 消费方式 | 人读 + 机读兼顾 | 6 |
| D2 | 顶层分类依据 | 按技术栈分层 | 3 |
| D3 | 文档与代码关系 | 讲解内嵌片段 + 完整代码同目录 `code/` | 5 |
| D4 | 深度强制维度 | 执行映射、数学/算法推导、踩坑与失败模式 | 7.2 |
| D5 | 性能实测数据 | **不强制**每篇覆盖 | 7.1 |
| D6 | 首期交付范围 | 骨架 + 一条完整样板链 | 8 |
| D7 | 样板链选型 | GPU 硬件，两条链共 6 篇 | 8.3 |
| D8 | 分类粒度 | 拆到 16 个一级域 | 3 |
| D9 | 旧项目处置 | 不处理，知识库自包含 | 1.2 |
| D10 | CUDA 执行环境 | 远程 Linux 机器，本机不编译 | 1.3 |
| D11 | 进度跟踪 | 全库进度表，简单易维护 | 9 |

---

## 3. 顶层分类体系

### 3.1 设计原则

- **按"栈的层"切分，不按技术点切分**。同一层内的内容共享前提知识，跨层内容用链接而非拷贝。
- **编号三位数，间隔 10**。插入新域不需重排，字典序与逻辑序一致。
- **一级域之下按主题分二级目录，必要时再分三级**。层级不超过三层，超过三层说明该域需要重新划分。

### 3.2 完整目录结构

标注说明：`[首期]` = 首期有实质内容；`[占位]` = 首期只有 README；`[留白]` = 首期不建，内容出现时再建。

```
knowledge/
├── README.md                    总入口：定位 + 学习路径图 + 导航
├── CONTRIBUTING.md              写作规范：命名 / frontmatter / 深度标准 / 链接约定
├── _template.md                 单篇文档模板（可直接复制）
├── .gitignore
├── docs/                        设计与规划工件，不属于知识内容
│   └── superpowers/specs/       本设计文档的存放位置
│
├── 00-meta/                     知识库自身：索引、术语、进度
│   ├── README.md                [首期] 本域说明
│   ├── glossary.md              [首期] 术语表中英对照
│   ├── index.md                 [首期] 全库索引，按域组织
│   └── roadmap.md               [首期] 全库进度表，见第 9 节
│
├── 10-foundations/              数学与数值基础
│   ├── floating-point/          IEEE754 格式、舍入模式、误差累积、次正规数
│   ├── numerical-linear-algebra/ 矩阵计算、条件数、数值稳定性、迭代求解
│   └── performance-models/      roofline 模型、算术强度、Amdahl 定律、延迟-带宽模型
│
├── 20-gpu-architecture/         GPU 体系结构
│   ├── execution-model/         [首期] SM 内部结构 / warp / SIMT / 调度器 / 分支发散
│   ├── memory-system/           [首期] 存储层次 / 访存合并 / 共享内存 bank 冲突
│   ├── tensor-core/             MMA 指令族 / wmma / wgmma / tcgen05 / 稀疏 MMA
│   ├── interconnect/            NVLink / NVSwitch / PCIe / 多卡拓扑与带宽
│   └── architecture-evolution/  Volta→Ampere→Hopper→Blackwell 逐代规格差异
│
├── 30-gpu-programming/          GPU 编程模型
│   ├── cuda/                    线程组织 / 同步原语 / stream / 内存管理 / CUDA Graph
│   ├── triton/                  Triton 语言模型 / 编译流程 / 常见 kernel 模式
│   ├── ptx-sass/                PTX 与 SASS 指令集 / 寄存器分配 / 指令调度 / 反汇编方法
│   └── cutlass/                 CUTLASS 模板抽象 / 层次化 tiling / collective 设计
│
├── 40-kernels/                  算子内核工程
│   ├── gemm/                    朴素 → tiling → 寄存器分块 → tcgen05 → persistent kernel
│   ├── attention/               flash attention 系列演进 / 变体 / 解码阶段 kernel
│   ├── reduce-scan/             归约与扫描的并行算法 / warp shuffle / 多级归约
│   ├── softmax-normalization/   softmax 数值稳定实现 / 在线 softmax / norm 融合
│   ├── elementwise-fusion/      逐元素算子融合 / 访存受限算子的优化边界
│   └── moe-kernels/             grouped GEMM / all-to-all / 专家调度 kernel
│
├── 50-performance/              性能工程（跨层，故独立成域）
│   ├── profiling-tools/         nsight compute / nsight systems / CUPTI / torch profiler
│   ├── bottleneck-diagnosis/    瓶颈定位方法论 / 常见反模式 / 计算-访存-延迟三分法
│   └── tuning-practice/         调优实战 / 参数搜索 / 可复现的基准测试方法
│
├── 60-compilers/                编译器与图优化
│   ├── mlir/                    MLIR 方言体系 / lowering 流程 / pass 编写
│   ├── tvm/                     TVM 与 TileLang / 调度原语 / 自动代码生成
│   ├── inductor/                torch.compile 后端 / 图捕获 / codegen / 与 Triton 的关系
│   ├── xla/                     TensorFlow 与 JAX 的图编译 / HLO / 算子融合
│   └── auto-tuning/             自动调优方法 / 搜索空间设计 / 代价模型
│
├── 70-frameworks/               深度学习框架
│   ├── pytorch/                 autograd / dispatch 机制 / aten / 自定义算子 / torch.compile
│   ├── tensorflow/              graph 与 eager / XLA 集成 / custom op / SavedModel
│   └── jax/                     jit / vmap / pmap / sharding / 与 PyTorch 的心智差异
│
├── 80-model-architectures/      模型架构（优化的对象）
│   ├── transformer/             原版结构 → 现代变体 / 残差设计 / 深度与宽度缩放
│   ├── attention-variants/      MHA / MQA / GQA / MLA / 线性注意力 / 混合注意力
│   ├── moe/                     gating 机制 / 专家路由 / 负载均衡 / 容量因子
│   ├── normalization/           LayerNorm / RMSNorm / QK-norm / pre-post 位置之争
│   └── positional-encoding/     RoPE / ALiBi / 相对位置 / 长度外推方法
│
├── 90-training/                 训练系统
│   ├── autograd-and-backward/   计算图构造 / 反向传播实现 / 梯度检查点
│   ├── mixed-precision/         AMP / loss scaling / fp8 训练 / 精度对齐与复现
│   ├── optimizers/              SGD / Adam / AdamW / Muon / 分布式优化器状态
│   ├── initialization-and-stability/  参数初始化 / 梯度裁剪 / loss spike 归因
│   └── checkpointing/           分片保存与恢复 / 容错 / 异步落盘
│
├── 100-parallelism/             并行策略
│   ├── data-parallel/           DP / DDP / 梯度规约（正文在 110，此处回链）
│   ├── tensor-parallel/         TP / 列行并行 / 通信量推导 / 与 TP 组划分
│   ├── pipeline-parallel/       PP / 1F1B / 气泡率分析 / 显存均衡
│   ├── sequence-context-parallel/  SP / CP / ring attention / 长序列切分
│   ├── expert-parallel/         EP / all-to-all 通信模式
│   └── zero-fsdp/               ZeRO 三阶段 / FSDP / 显存占用核算
│
├── 110-communication/           通信与网络
│   ├── collectives/             allreduce / allgather / reducescatter 算法与复杂度
│   ├── nccl/                    NCCL 架构 / 调优 / 拓扑感知 / 常见报错归因
│   ├── network-topology/        InfiniBand / RoCE / NVLink 域 / 胖树拓扑
│   └── overlap/                 通信-计算重叠 / 双缓冲 / 分块流水
│
├── 120-inference/               推理引擎与部署
│   ├── kv-cache/                KV cache 原理与显存核算 / PagedAttention / 前缀共享
│   ├── batching-and-scheduling/ continuous batching / chunked prefill / 调度策略
│   ├── decoding-strategies/     beam search / sampling / speculative decoding / 并行解码
│   ├── engines/                 vLLM / TensorRT-LLM / SGLang / llama.cpp 架构对比
│   └── serving-infra/           部署形态 / PD 分离 / 扩缩容 / 可观测性
│
├── 130-quantization/            量化与压缩
│   ├── numerical-formats/       fp16 / bf16 / tf32 / fp8 / int8 / int4 / nvfp4
│   ├── theory/                  量化误差分析 / scale 与 zero-point / outlier 问题
│   ├── ptq/                     训练后量化 / 校准方法 / GPTQ / AWQ / SmoothQuant
│   ├── qat/                     量化感知训练 / 伪量化 / 梯度近似
│   ├── quantized-kernels/       量化算子实现（正文在本域，回链 40-kernels）
│   ├── kv-cache-quantization/   KV cache 量化（正文在本域，回链 120-inference）
│   └── sparsity-and-distillation/  结构化与非结构化稀疏 / 知识蒸馏
│
├── 140-papers/                  论文精读
│   └── （无固定二级结构，按论文名建文件）
│
└── 150-playground/              实验场
    └── （无固定二级结构，临时实验与验证脚本）
```

### 3.3 关键划分决策

**优化器不独立成一级域**，保留在 `90-training/optimizers/`。理由：优化器与训练循环、混合精度、分布式优化器状态强耦合，拆出后 `90-training` 会失去核心内容。

**`50-performance` 独立成一级域**。理由：性能工程横跨硬件、框架、引擎三层，方法论只应在一处讲清楚，其余位置回链。

**模型架构置于 `80-model-architectures`**，位于框架与训练之间。理由：它是上游一切优化工作的对象，需先于训练/推理理解。

**`140-papers` 与 `150-playground` 是两个无固定结构的域**：

- `140-papers` 收录**论文本身的精读**（动机、方法、实验、局限）。若某篇论文的结论已沉淀为某个主题的正文（如 flash attention 的算法在 `40-kernels/attention/`），则**正文仍归该主题**，`140-papers` 只保留该论文的独立分析并双向 `related` 链接。
- `150-playground` 存放**尚未归位的实验**。一旦某实验沉淀为可复用的知识，须迁入对应域，**不在 playground 长期留存**。

### 3.4 与其他域的引用注记规范

凡正文不在本目录的主题，须在目录说明中以 `（正文在 X，此处回链）` 标注。第 3.2 节已对 `130-quantization/quantized-kernels`、`130-quantization/kv-cache-quantization`、`100-parallelism/data-parallel` 三处标注；后续新增同类情况照此处理。

---

## 4. 横切内容的处理规则

量化、通信、性能、并行天然跨越多个栈层。规则如下：

1. **一个知识点只有一处正文**，位置由"该知识的核心归属层"决定。
2. 其他位置**只放链接**，不复制内容。
3. frontmatter 的 `related` 字段承载跨域引用，保证机器检索时能重建关联。

示例：

| 知识点 | 正文位置 | 引用位置 |
|---|---|---|
| block-wise 量化 kernel 实现 | `130-quantization/quantized-kernels/` | `40-kernels/gemm/` |
| NVLink 带宽与拓扑 | `20-gpu-architecture/interconnect/` | `110-communication/nccl/` |
| roofline 模型 | `10-foundations/performance-models/` | 所有 kernel 文档 |
| 梯度规约算法 | `110-communication/collectives/` | `100-parallelism/data-parallel/` |

**判定依据**：如果去掉该层，这个知识点还成立吗？成立则归更基础的层。

**链接有效性**：允许 `related` / `prerequisites` 指向**尚未创建**的文档路径。这类链接是知识图谱的前向声明，不视为错误。链接有效性检查列为后续工作（见 10），首期不做。

---

## 5. 文档与代码的组织机制

### 5.1 编号即外键

文档编号与代码编号**一一对应**，这是防止两者脱钩的核心机制。对应关系是：`NN-<文档名>.md` ↔ `code/NN_<代码名>.<ext>`，其中 `NN` 相同。

```
20-gpu-architecture/memory-system/
├── 01-hierarchy-overview.md            ←→  code/01_bandwidth_probe.cu
├── 02-coalescing-and-cache-line.md     ←→  code/02_coalescing_demo.cu
├── 03-shared-memory-bank-conflict.md   ←→  code/03_bank_conflict_demo.cu
└── code/
    ├── 01_bandwidth_probe.cu
    ├── 02_coalescing_demo.cu
    ├── 03_bank_conflict_demo.cu
    └── Makefile
```

**约束**：一个主题目录下，文档数与 `code/` 中的源文件数必须相等，编号一一对应。**没有配套代码的文档不单独存在**——若某篇文档确实不需要代码，则该主题下所有文档都不建 `code/`（见 5.3）。不允许出现"3 篇文档配 2 个代码文件"的错位。

### 5.2 分工

- **正文中的代码**：只嵌入讲解所需的**关键片段**，保证阅读连贯，不必来回跳转。
- **`code/` 目录中的代码**：**完整、可编译、可运行**的版本，每篇文档末尾给出编译运行命令。
- **`code/` 必须自包含**：自带 `Makefile`，不依赖仓库中其他目录，可单独拷贝到远程机器运行。

### 5.3 非代码型主题

对于 `90-training`、`120-inference` 等不以源码为中心的主题，`code/` 目录**可选**：

- 若提供，内容是**可运行的验证脚本或配置**（启动命令、配置片段、对比实验脚本），仍须遵守 5.1 的编号对应。
- 若不提供，该主题目录下**不建 `code/`**，正文不出现指向 `code/` 的链接。

---

## 6. frontmatter 规范

每篇文档头部必须包含以下 YAML frontmatter。它承载全部机器可读元数据，且不影响 Markdown 渲染。

```yaml
---
title: 共享内存 bank 冲突
slug: shared-memory-bank-conflict
domain: 20-gpu-architecture
topic: memory-system
level: 2
difficulty: intermediate
prerequisites:
  - 20-gpu-architecture/memory-system/02-coalescing-and-cache-line
tags: [shared-memory, bank-conflict, padding]
related:
  - 40-kernels/gemm/02-shared-memory-tiling
code:
  - code/03_bank_conflict_demo.cu
status: draft
updated: 2026-09-14
---
```

### 字段说明

| 字段 | 必填 | 类型 | 取值与约定 |
|---|---|---|---|
| `title` | 是 | string | 文档标题，与正文一级标题一致 |
| `slug` | 是 | string | 全局唯一，kebab-case，用于跨文档引用 |
| `domain` | 是 | string | 一级域**目录名全称**，含编号，如 `20-gpu-architecture` |
| `topic` | 否 | string | 二级目录名；无二级目录时**省略该字段**（不写空串、不写 null） |
| `level` | 是 | int | `1`=概念 · `2`=原理 · `3`=实现 · `4`=优化 · `5`=前沿 |
| `difficulty` | 是 | enum | `beginner` · `intermediate` · `advanced` |
| `prerequisites` | 是 | list | 前置文档的**仓库根相对路径**（不含 `.md` 后缀）；无前置时写 `[]` |
| `tags` | 是 | list | 检索标签，小写 kebab-case；无标签时写 `[]` |
| `related` | 是 | list | 相关文档路径，格式同 `prerequisites`；可跨域；无则 `[]` |
| `code` | 否 | list | **文档所在目录的相对路径**，如 `code/03_bank_conflict_demo.cu`；无代码时不写该字段 |
| `status` | 是 | enum | `draft` · `stable` · `deprecated`，判定见 7.2 |
| `updated` | 是 | date | 最后更新日期，格式 `YYYY-MM-DD` |

**路径基准**：`prerequisites` 与 `related` 用**仓库根相对路径**（跨域引用必须如此）；`code` 用**文档所在目录的相对路径**（因为每个主题的 `code/` 自成一体）。

**`prerequisites` 与 `related` 的区别**：前者是**必须**先读的（否则看不懂），后者是**建议**延伸的。这个区分是知识图谱的边类型，检索时需要区别对待。

**`prerequisites` 构成有向无环图**。新增文档时须人工核对不引入环，核对清单见 8.4。

### status 判定

| 取值 | 判定标准 |
|---|---|
| `draft` | 第 0–4 节**未全部覆盖**，或未经一次校对 |
| `stable` | 第 0–4 节全部覆盖，且通过 7.2 的自查清单 |
| `deprecated` | 内容已过时，文中已注明替代文档 |

**`status` 与第 5 节实测数据无关**——第 5 节是可选章节，缺失不影响 `stable` 判定。

---

## 7. 文档模板与深度标准

### 7.1 模板

```markdown
---
（frontmatter 见第 6 节）
---

# 标题

> 一句话结论。读完这句就知道全文要讲什么。

## 0. 背景与动机

这项技术解决什么问题？不用它的代价是什么？朴素做法的瓶颈在哪？
须给出一个**具体的对照场景**，而非泛泛而谈。

## 1. 原理

数学/算法层面的推导。公式怎么来的、复杂度如何、边界条件是什么。
这一节应当让读者能自己重新推导出结论，而不是记住结论。

## 2. 执行映射

这段东西在真实系统上实际发生了什么。按主题层次选择切入点：

- **硬件层主题**：在 SM、warp、寄存器、各级缓存、互联上发生了什么。
- **框架层主题**：在 dispatch、计算图、调度器、内存分配器上发生了什么。
- **引擎层主题**：在调度策略、batching、显存管理上发生了什么。

这一节回答"为什么这个写法快、那个写法慢"。

## 3. 实现与演进

朴素做法 → 主流实现 → 前沿方案。每一步说明**为什么需要下一步**。
关键代码/配置片段内嵌在正文中，完整可运行版本见 `code/`（如有）。

## 4. 踩坑与失败模式

错误直觉、性能退化条件、数值陷阱、版本差异、常见误用。
每一条须写成「**现象 → 成因 → 规避**」三段式。

## 5. 实测数据

**可选章节**。只放数字与测量方法，并注明 GPU 型号与 CUDA 版本。
未实测的数字须标注为「推导值」。测量方法本身链接到 `50-performance/`。

## 6. 小结与延伸

要点回顾 + 指向 `related` 中的文档。

## 参考

官方文档、论文、源码位置。
```

### 7.2 深度标准与自查清单

一篇文档算"写完"（`status: stable`），须通过以下**五条自查**。前四条可机械核对，第五条需一次通读。

| # | 检查项 | 通过判据 |
|---|---|---|
| 1 | **章节完整** | 第 0、1、2、3、4 节标题均存在且非空 |
| 2 | **有推导过程** | 第 1 节包含至少一处**多步推导**（含中间步骤的公式变形或复杂度分析），而非仅罗列结论公式 |
| 3 | **有系统细节** | 第 2 节至少指明 **2 个具体组件名**（如 warp scheduler、L1 sector、dispatch key）并给出**至少 1 个数值量级**（如延迟 cycles、带宽 GB/s、寄存器数） |
| 4 | **有失败模式** | 第 4 节至少 1 条，且为「现象 → 成因 → 规避」完整三段式 |
| 5 | **可独立读懂** | 通读一遍：不依赖未列在 `prerequisites` 中的文档即可理解 |

第 2、3 条把主观判断转化为**可数判据**（是否有中间步骤、组件名个数、数值量级个数），便于核对，也便于将来脚本化。

### 7.3 篇幅指引

- 单个文档聚焦**一个概念**，不追求大而全。超过约 500 行说明该拆分。
- 宁可拆成两篇并建立 `prerequisites` 关系，也不要写一篇长文。

---

## 8. 命名规范

| 对象 | 规范 | 示例 |
|---|---|---|
| 一级域目录 | `NNN-kebab-case`，三位数，间隔 10 | `20-gpu-architecture` |
| 二级/三级目录 | `kebab-case`，无编号 | `memory-system` |
| 文档文件 | `NN-kebab-case.md`，两位数，域内主题内唯一 | `02-coalescing-and-cache-line.md` |
| 代码文件 | `NN_snake_case.ext`，`NN` 与对应文档相同 | `02_coalescing_demo.cu` |
| 目录索引 | 固定名 `README.md` | — |
| 非知识文档 | 前缀下划线 `_`，表示"不是知识内容" | `_template.md` |
| 设计与规划工件 | 放 `docs/`，不进 16 个一级域 | `docs/superpowers/specs/` |

**关于下划线前缀**：含义是"**非知识文档**"（模板、临时文件），而非"草稿"。知识内容一律不加下划线前缀。

### 8.4 新增文档核对清单

每次新增或修改文档，人工核对以下四项：

1. `prerequisites` 中的所有路径是否已存在，或是否为**有意的前向声明**？
2. 新增的 `prerequisites` 边是否引入**环**？（沿 `prerequisites` 向上遍历能否回到自身）
3. `code` 中的编号是否与文档编号一致，且文件确实存在？
4. `status` 是否与 7.2 的判定相符？

---

## 9. 进度表（`00-meta/roadmap.md`）

### 9.1 设计要求

- **简单明了、易于维护**：新增一篇文档时，改动量不超过一行。
- **覆盖全库**：不限于首期，所有一级域的所有二级主题都在表内。
- **机器可解析**：使用标准 Markdown 任务列表语法，便于将来脚本统计。

### 9.2 格式

粒度到**二级主题目录**（不细到单篇文档）。状态用三档：

- `- [x]` 完成：该主题下的文档均满足 7.2 的深度标准
- `- [ ]` 未开始
- `- [ ] … ← 进行中` 进行中

```markdown
# 实施进度表

> 最后更新：2026-09-14
> 进度：6 / 61 主题完成

## 10-foundations
- [ ] floating-point
- [ ] numerical-linear-algebra
- [ ] performance-models

## 20-gpu-architecture
- [x] execution-model
- [x] memory-system
- [ ] tensor-core
- [ ] interconnect
- [ ] architecture-evolution

（其余域同理，每个一级域一个二级标题）

## 140-papers
- [ ] （无固定结构，按需追加）

## 150-playground
- [ ] （无固定结构，按需追加）
```

### 9.3 维护规则

1. **主题状态变化即更新**：某主题从"未开始"变"进行中"或"完成"时，改动对应那一行。
2. **顶部两行同步更新**：`最后更新` 与 `进度: X / Y 主题完成`。
3. **新增主题**：在所属域下追加一行 `- [ ] <topic>`，并同步更新总数。
4. **不在进度表中记录单篇文档**，避免维护负担。

---

## 10. 首期交付范围

首期工作分两组，性质不同，**分别验收**。

### 10.1 第一组：仓库基础与骨架（机械性工作）

| # | 交付物 | 说明 |
|---|---|---|
| A1 | `.gitignore` | 已存在，确认内容合适即可 |
| A2 | `README.md` | 知识库定位、16 个一级域说明、学习路径图、如何使用 |
| A3 | `CONTRIBUTING.md` | 命名规范、frontmatter 规范、深度标准与自查清单、链接约定、新增文档核对清单 |
| A4 | `_template.md` | 第 7.1 节模板的可复制版本 |
| A5 | 目录骨架 | 第 3.2 节中标注 `[首期]` 与 `[占位]` 的全部目录 |
| A6 | 一级域 README | **16 个**一级域各一份，内容：本域定位、计划覆盖的主题清单、与其他域的关系。二级目录**不建 README**，待有内容时再建。 |
| A7 | `00-meta/glossary.md` | 术语表中英对照，**不少于 30 条**，首批覆盖 GPU 与量化核心术语 |
| A8 | `00-meta/index.md` | 全库索引，按域组织，列出已建目录 |
| A9 | `00-meta/roadmap.md` | 第 9 节格式的完整进度表，含全部 61 个二级主题 |

**第一组验收标准**

1. 目录结构与第 3.2 节标注 `[首期]` / `[占位]` 的部分一致（`docs/` 与根目录文件除外的额外目录为空）。
2. 16 个一级域均有 README，且包含"域定位 / 计划主题 / 与其他域关系"三部分。
3. `roadmap.md` 中的主题总数与第 3.2 节的二级目录数一致。
4. `CONTRIBUTING.md` 与第 6、7、8 节内容一致，无矛盾。
5. `glossary.md` 条数 ≥ 30。

### 10.2 第二组：样板链 `20-gpu-architecture/`（高智力密度工作）

**六篇文档**，编号与 `code/` 中六个源文件一一对应。

#### `execution-model/`

| 文档 | 0. 背景与动机 | 1. 原理（多步推导） | 2. 执行映射（组件+数值） | 3. 实现与演进 | 4. 踩坑与失败模式 |
|---|---|---|---|---|---|
| `01-sm-internals.md` | 为什么"我有多少并行度"这个问题无法脱离 SM 结构回答；朴素认知"核心数=并行度"为何错 | 寄存器文件是硬预算：`寄存器文件大小 ÷ (每线程寄存器数 × 32) = 可常驻 warp 数上限`，推导寄存器约束方程 | 组件：CUDA core / warp scheduler / register file / LSU / SFU；数值：历代寄存器文件大小、每 SM 最大常驻 warp 数 | Volta→Turing→Ampere→Hopper→Blackwell 的 SM 规格演进，每代变化的动因 | 跨代 SM 直接比数量；把寄存器文件当缓存而非预算；`__launch_bounds__` 误用 |
| `02-simt-and-divergence.md` | SIMT 让代码写起来像标量程序，但硬件是向量化的；发散"免费"是假象 | SIMT vs SIMD 的本质区别（谁负责向量化）；有效并行度模型：warp 内 N 条互斥路径，最坏串行化后有效并行度 = 32/N | 组件：warp 内 32 线程锁步、reconvergence stack、SASS 的 `BRA`/`BSSY`/`BSYNC`、predication；数值：分支发散的周期代价量级 | 从分支判断到无分支写法的几种手段：算术选择、查表、warp 级原语（`__shfl`、`__ballot_sync`）、Volta 独立线程调度的影响 | 以为发散只来自 `if/else`（循环边界不齐同样发散）；`__syncwarp` 语义误解；跨代独立线程调度差异 |
| `03-occupancy-and-latency-hiding.md` | GPU 开那么多线程不是为了并行计算，是为了隐藏延迟 | Little's Law 推导：所需并发度 = 延迟 ÷ 吞吐；占用率约束方程（寄存器、共享内存、block 数、warp 数四者取最小） | 组件：warp 切换机制、ILP 与 TLP 的分工、访存延迟链路；数值：各级访存延迟的 cycles 量级 | `cudaOccupancyMaxActiveBlocksPerMultiprocessor` API；`__launch_bounds__` 与 `maxrregcount` 的取舍 | 占用率不是越高越好（高占用反而慢的真实成因）；把 occupancy 当唯一指标 |

#### `memory-system/`

| 文档 | 0. 背景与动机 | 1. 原理（多步推导） | 2. 执行映射（组件+数值） | 3. 实现与演进 | 4. 踩坑与失败模式 |
|---|---|---|---|---|---|
| `01-hierarchy-overview.md` | 为什么存储必须分级；CPU 与 GPU 存储层次的取向差异（CPU 靠大缓存，GPU 靠大量线程隐藏延迟） | 各级存储容量/带宽/延迟的量级对比与推导；带宽与延迟的物理约束（信号传播距离、功耗）；共享内存为何是"软件管理的缓存" | 组件：寄存器 → 共享内存 → L1 → L2 → HBM 的数据路径、local memory 的物理归属；数值：各级容量/带宽/延迟量级表 | 从统一内存到 HBM3e 的演进；Hopper/Blackwell 的 L2 分区与 cluster 对存储层次的影响 | 把共享内存当通用缓存；忽略 L2 容量导致命中率崩塌；混淆 local memory 与 global memory |
| `02-coalescing-and-cache-line.md` | 同样的计算量，改变访存顺序性能差数十倍 | 事务数推导：32 线程 × 4B 连续 = 1 个 128B 事务；跨步访问 = 32 个事务；有效带宽 = 有用字节 ÷ 传输字节 | 组件：cache line（128B）、L1 sector（32B）、L2 的 sector 访问、合并判定规则、写策略；数值：不同访问模式的事务数与有效带宽 | SoA vs AoS 的取舍；向量化访问（`float4`）；`__ldg` 与只读缓存；Hopper TMA 对访存模式的改变 | AoS vs SoA 的真实代价（并非总是 SoA 好）；以为"地址连续"就等于"合并"（关键是 warp 内线程到数据的映射）；把只读缓存当万能加速 |
| `03-shared-memory-bank-conflict.md` | 共享内存比全局快近百倍，但用错了可以比全局还慢 | bank 寻址模型 `bank = (addr / 4) % 32`；一个 warp 内访问的冲突度计算；broadcast 与多播的免费情形 | 组件：32 个 bank、每 bank 4B 宽、冲突时的串行化、64-bit 访问的成对机制、padding 对 bank 映射的改变；数值：冲突度与访问周期数的对应 | padding（`[32][33]`）的推导；swizzle 技巧；Hopper 分布式共享内存（DSMEM / cluster） | padding 方向搞反；以为加 padding 总是有益（牺牲容量与合并）；以为 64-bit 访问没有冲突 |

**六个代码文件**

```
20-gpu-architecture/execution-model/code/
├── 01_sm_probe.cu           查询 device 属性：SM 数、寄存器文件、每 SM 最大 warp 数
├── 02_divergence_demo.cu    不同分支发散比例的 kernel 计时对比
├── 03_occupancy_probe.cu    occupancy API 实测 + 占用率与吞吐的关系曲线
└── Makefile

20-gpu-architecture/memory-system/code/
├── 01_bandwidth_probe.cu    各级存储带宽实测
├── 02_coalescing_demo.cu    合并 vs 跨步访存对比
├── 03_bank_conflict_demo.cu 有/无 padding 的共享内存访问对比
└── Makefile
```

代码要求：

- `nvcc` 可直接编译，零 error。
- 运行时查询 device 属性选择路径，不硬编码架构。
- 输出可直接验证文档中的结论；注释说明观测要点。
- 文档第 5 节的实测数字，若在远程机器跑过则注明 GPU 型号与 CUDA 版本；未跑则标注「推导值」。

**第二组验收标准**

1. 第 3.2 节中 `execution-model/` 与 `memory-system/` 标注 `[首期]` 的部分结构与实际一致。
2. 6 篇文档均通过第 7.2 节的五条自查，`status` 为 `stable`。
3. 6 个 `.cu` 文件编号与文档一一对应，在远程 Linux 机器上 `make` 零 error。
4. 每篇文档末尾有可执行的编译运行命令。
5. 文档中每一个实测数字都有对应的测量来源标注（GPU 型号 + CUDA 版本），或明确标注「推导值」。
6. `00-meta/roadmap.md` 中 `20-gpu-architecture` 下的 `execution-model` 与 `memory-system` 两行标记为完成。

### 10.3 首期不做

- `20-gpu-architecture/` 的其余三个二级目录（tensor-core / interconnect / architecture-evolution）只放目录，内容留待后续
- 其余 15 个一级域只有 README，无实质内容
- 索引生成脚本、进度自动统计脚本
- 静态站点
- 二级目录 README

---

## 11. 后续演进

以下内容不在首期，但方案已为其预留接口：

- **索引自动化**：当 `00-meta/index.md` 的手工维护成为负担时（触发条件：文档数 > 50），引入脚本扫描 frontmatter 生成索引与进度统计。
- **链接有效性检查**：同样在文档数 > 50 时引入，检查 `prerequisites` / `related` 指向的路径是否存在、是否成环。
- **向量化检索**：`slug` / `domain` / `tags` / `prerequisites` 已构成检索元数据；单文件单概念的切分已保证 chunk 质量。
- **内容填充**：按 `00-meta/roadmap.md` 的顺序，以首期 6 篇为质量标尺逐域填充。

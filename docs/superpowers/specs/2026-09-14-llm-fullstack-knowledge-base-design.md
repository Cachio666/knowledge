# 大模型全栈知识库 设计方案

- **日期**：2026-09-14
- **状态**：待实施
- **目标仓库**：`/Users/cachio/gitProjects/CUDA/knowledge`

---

## 1. 背景与目标

### 1.1 背景

需要一个用于学习的技术知识库。学习目标是**大模型从硬件底层到引擎上层的完整链路**，涵盖训练与推理的全部内容。最初的切入点是 CUDA 算子开发、模型量化、GPU 编程、PyTorch、TensorFlow 五项，但经确认这五项只是入口，知识库的边界应是**整个大模型技术栈**。

仓库当前为空目录，非 git 仓库。同级存在 `cuda-learning`、`CUDA_Freshman`、`cutlass`、`Pytorch-Inductor-Tutorial` 等学习项目。

### 1.2 目标

1. 建立覆盖大模型全栈的**可扩展**目录体系，新领域可直接插入而不触发现有结构。
2. 文档**深入原理**，而非资料摘抄。深度由可检查的维度标准定义。
3. 文档与代码**明确区分**，且两者保持可追溯的对应关系。
4. 全部采用 Markdown 格式。
5. 同时服务于**人阅读**（连贯、可导航）与**机器检索**（结构化元数据、语义边界清晰）。

### 1.3 非目标（YAGNI）

以下内容明确不在本方案范围内：

- **不做静态站点生成**（VitePress / MkDocs 等）。当前阶段纯 Markdown + 编辑器/文件系统导航即可，避免引入构建链。
- **不做向量化入库与检索服务**。frontmatter 与文件切分规则为将来留好接口，但现在不实现。
- **不迁移或拆分现有同级项目**。`cuda-learning`、`CUDA_Freshman`、`Pytorch-Inductor-Tutorial` 保持原状，知识库自包含、不依赖它们。
- **不写索引生成脚本**。首期索引文件手工维护；待文档量增长到手工维护成为负担时再引入脚本。
- **不追求内容全覆盖**。首期只交付骨架与一条完整的样板链，其余目录放 README 占位。

---

## 2. 需求决策记录

以下决策经逐项确认，构成本方案的约束基础：

| 编号 | 决策点 | 结论 | 影响 |
|---|---|---|---|
| D1 | 消费方式 | 人读 + 机读兼顾 | 需要 frontmatter 元数据；文件按单一概念切分 |
| D2 | 顶层分类依据 | 按技术栈分层 | 从硬件到应用自下而上编号 |
| D3 | 文档与代码关系 | 讲解内嵌片段 + 完整代码同目录 `code/` | 见第 5 节 |
| D4 | 深度强制维度 | 硬件/执行映射、数学/算法推导、踩坑与失败模式 | 见第 7.2 节 |
| D5 | 性能实测数据 | **不强制**每篇覆盖 | 归入 `50-performance`，正文按需回链 |
| D6 | 首期交付范围 | 骨架 + 一条完整样板链 | 见第 9 节 |
| D7 | 样板链选型 | GPU 硬件，两条链共 6 篇 | execution-model 3 篇 + memory-system 3 篇 |
| D8 | 分类粒度 | 拆到 16 个一级域 | 见第 3 节 |
| D9 | 旧项目处置 | 不处理，知识库自包含 | 代码示例全部重写 |

---

## 3. 顶层分类体系

### 3.1 设计原则

- **按"栈的层"切分，不按技术点切分**。同一层内的内容共享前提知识，跨层内容用链接而非拷贝。
- **编号三位数，间隔 10**。插入新域不需重排，字典序与逻辑序一致。
- **一级域之下按主题分二级目录，必要时再分三级**。层级不超过三层，超过三层说明该域需要重新划分。

### 3.2 完整目录结构

```
knowledge/
├── README.md                    总入口：定位 + 学习路径图 + 导航
├── CONTRIBUTING.md              写作规范：命名 / frontmatter / 深度标准 / 链接约定
├── _template.md                 单篇文档模板
├── _meta/
│   ├── glossary.md              术语表（中英对照，全库统一口径）
│   ├── index.md                 全库索引
│   └── roadmap.md               待写清单与进度
│
├── 00-meta/                     知识库自身：规范、模板、索引、术语
│
├── 10-foundations/              数学与数值基础
│   ├── floating-point/          IEEE754、舍入模式、误差累积、次正规数
│   ├── numerical-linear-algebra/ 矩阵计算、条件数、数值稳定性
│   └── performance-models/      roofline、算术强度、Amdahl、延迟-带宽模型
│
├── 20-gpu-architecture/         GPU 体系结构
│   ├── execution-model/         SM / warp / SIMT / 调度器 / 分支发散
│   ├── memory-system/           寄存器→smem→L1/L2→HBM / 合并 / bank
│   ├── tensor-core/             MMA / wgmma / tcgen05 / 稀疏 MMA
│   ├── interconnect/            NVLink / NVSwitch / PCIe / 多卡拓扑
│   └── architecture-evolution/  逐代架构差异（Volta→Ampere→Hopper→Blackwell）
│
├── 30-gpu-programming/          GPU 编程模型
│   ├── cuda/                    线程组织 / 同步 / stream / 内存管理 / CUDA Graph
│   ├── triton/                  语言模型 / 编译流程 / 常见模式
│   ├── ptx-sass/                指令集 / 寄存器分配 / 指令调度 / 反汇编
│   └── cutlass/                 模板抽象 / 层次化 tiling / collective
│
├── 40-kernels/                  算子内核工程
│   ├── gemm/                    朴素→tiling→寄存器→tcgen05→persistent
│   ├── attention/               flash attention 系列、变体、解码 kernel
│   ├── reduce-scan/
│   ├── softmax-normalization/
│   ├── elementwise-fusion/
│   └── moe-kernels/             grouped GEMM / all-to-all / 专家调度
│
├── 50-performance/              性能工程（跨层，故独立成域）
│   ├── profiling-tools/         nsight compute / nsight systems / CUPTI / torch profiler
│   ├── bottleneck-diagnosis/    瓶颈定位方法论、常见反模式
│   └── tuning-practice/         调优实战、参数搜索
│
├── 60-compilers/                编译器与图优化
│   ├── mlir/                    方言体系 / lowering / pass
│   ├── tvm/                     TVM / TileLang / 调度原语
│   ├── inductor/                torch.compile 后端 / codegen
│   ├── xla/                     TF / JAX 图编译
│   └── auto-tuning/
│
├── 70-frameworks/               深度学习框架
│   ├── pytorch/                 autograd / dispatch / aten / 自定义算子 / compile
│   ├── tensorflow/              graph & eager / XLA / custom op
│   └── jax/                     jit / vmap / sharding
│
├── 80-model-architectures/      模型架构（优化的对象）
│   ├── transformer/             原版→现代变体 / 残差 / 深度缩放
│   ├── attention-variants/      MHA / MQA / GQA / MLA / 线性注意力
│   ├── moe/                     gating / 路由 / 负载均衡
│   ├── normalization/           LayerNorm / RMSNorm / QK-norm / pre-post 位置
│   └── positional-encoding/     RoPE / ALiBi / 长度外推
│
├── 90-training/                 训练系统
│   ├── autograd-and-backward/   计算图 / 反向实现 / 梯度检查点
│   ├── mixed-precision/         AMP / loss scaling / fp8 训练 / 精度对齐
│   ├── optimizers/              SGD / Adam / AdamW / Muon / 分布式优化器状态
│   ├── initialization-and-stability/  初始化 / 梯度裁剪 / loss spike 排查
│   └── checkpointing/           分片保存 / 恢复 / 容错
│
├── 100-parallelism/             并行策略
│   ├── data-parallel/           DP / DDP / 梯度规约
│   ├── tensor-parallel/         TP / 列行并行 / 通信量分析
│   ├── pipeline-parallel/       PP / 1F1B / 气泡分析 / 显存均衡
│   ├── sequence-context-parallel/  SP / CP / ring attention
│   ├── expert-parallel/         EP / all-to-all
│   └── zero-fsdp/               ZeRO 三阶段 / FSDP / 显存核算
│
├── 110-communication/           通信与网络
│   ├── collectives/             allreduce / allgather / reducescatter 算法与复杂度
│   ├── nccl/                    NCCL 架构 / 调优 / 拓扑感知
│   ├── network-topology/        InfiniBand / RoCE / NVLink 域 / 胖树
│   └── overlap/                 通信-计算重叠 / 双缓冲
│
├── 120-inference/               推理引擎与部署
│   ├── kv-cache/                原理 / PagedAttention / 显存核算
│   ├── batching-and-scheduling/ continuous batching / chunked prefill / 调度策略
│   ├── decoding-strategies/     beam / sampling / speculative decoding / 并行解码
│   ├── engines/                 vLLM / TensorRT-LLM / SGLang / llama.cpp
│   └── serving-infra/           部署 / PD 分离 / 扩缩容 / 可观测性
│
├── 130-quantization/            量化与压缩
│   ├── numerical-formats/       fp16 / bf16 / tf32 / fp8 / int8 / int4 / nvfp4
│   ├── theory/                  误差分析 / scale 与 zero-point / outlier
│   ├── ptq/                     训练后量化
│   ├── qat/                     量化感知训练
│   ├── quantized-kernels/       量化算子实现（正文在本域，回链 40-kernels）
│   ├── kv-cache-quantization/   KV cache 量化（正文在本域，回链 120-inference）
│   └── sparsity-and-distillation/
│
├── 140-papers/                  论文精读
│
└── 150-playground/              实验场
```

### 3.3 关键划分决策

**优化器不独立成一级域**，保留在 `90-training/optimizers/`。理由：优化器与训练循环、混合精度、分布式优化器状态强耦合，拆出后 `90-training` 会失去核心内容。

**`50-performance` 独立成一级域**。理由：性能工程横跨硬件、框架、引擎三层，方法论只应在一处讲清楚，其余位置回链。

**模型架构置于 `80-model-architectures`**，位于框架与训练之间。理由：它是上游一切优化工作的对象，需要先于训练/推理理解。

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

---

## 5. 文档与代码的组织机制

### 5.1 编号即外键

文档编号与代码编号一一对应，这是防止两者脱钩的核心机制：

```
20-gpu-architecture/memory-system/
├── 01-hierarchy-overview.md              ←→  code/01_bandwidth_probe.cu
├── 02-coalescing-and-cache-line.md       ←→  code/02_coalescing_demo.cu
├── 03-shared-memory-bank-conflict.md     ←→  code/03_bank_conflict_demo.cu
└── code/
    ├── 01_bandwidth_probe.cu
    ├── 02_coalescing_demo.cu
    ├── 03_bank_conflict_demo.cu
    └── Makefile
```

### 5.2 分工

- **正文中的代码**：只嵌入讲解所需的**关键片段**，保证阅读连贯，不必来回跳转。
- **`code/` 目录中的代码**：**完整、可编译、可运行**的版本，每篇文档末尾给出编译运行命令。
- **`code/` 必须自包含**：自带 `Makefile`，不依赖仓库中其他目录，可单独拷贝出去运行。

### 5.3 非代码型主题

对于 `90-training`、`120-inference` 等不以源码为中心的主题，`code/` 目录可选。若提供，内容是**可运行的验证脚本或配置**（如启动命令、配置片段、对比实验脚本）。若无，则该主题目录下不建 `code/`。

---

## 6. frontmatter 规范

每篇文档头部必须包含以下 YAML frontmatter。它承载全部机器可读元数据，且不影响 Markdown 渲染。

```yaml
---
title: 共享内存 bank 冲突
slug: shared-memory-bank-conflict
domain: 20-gpu-architecture
topic: memory-system
level: 2                # 1=概念 2=原理 3=实现 4=优化 5=前沿
difficulty: intermediate  # beginner | intermediate | advanced
prerequisites:
  - 20-gpu-architecture/memory-system/02-coalescing-and-cache-line
tags: [shared-memory, bank-conflict, padding]
related:
  - 40-kernels/gemm/02-shared-memory-tiling
code:
  - code/03_bank_conflict_demo.cu
status: draft           # draft | stable | deprecated
updated: 2026-09-14
---
```

### 字段说明

| 字段 | 必填 | 说明 |
|---|---|---|
| `title` | 是 | 文档标题，与一级标题一致 |
| `slug` | 是 | 全局唯一标识，kebab-case，用于跨文档引用 |
| `domain` | 是 | 所属一级域目录名 |
| `topic` | 是 | 所属二级目录名；无二级目录时留空 |
| `level` | 是 | 认知层级，决定前置知识强度 |
| `difficulty` | 是 | 难度分级 |
| `prerequisites` | 是 | 前置文档路径数组，可为空数组 |
| `tags` | 是 | 检索标签数组 |
| `related` | 是 | 相关文档路径数组，可跨域 |
| `code` | 否 | 关联代码文件相对路径数组 |
| `status` | 是 | 文档状态，`draft` 表示尚未完成深度维度 |
| `updated` | 是 | 最后更新日期 |

**`prerequisites` 与 `related` 的区别**：前者是**必须**先读的（否则看不懂），后者是**建议**延伸的。这个区分是知识图谱的边类型，检索时需要区别对待。

**`prerequisites` 构成有向无环图**。新增文档时必须检查不引入环。

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

朴素做法 → 主流实现 → 前沿方案。关键代码/配置片段内嵌在正文中，
完整可运行版本见 `code/`（如有）。

## 4. 踩坑与失败模式

错误直觉、性能退化条件、数值陷阱、版本差异、常见误用。
这一节记录"看起来对但实际错"的情况。

## 5. 实测数据

**可选章节**。只放数字与测量方法。测量方法本身链接到 `50-performance/`。

## 6. 小结与延伸

要点回顾 + 指向 `related` 中的文档。

## 参考

官方文档、论文、源码位置。
```

### 7.2 深度标准

一篇文档算"写完"，必须同时满足：

1. **覆盖第 0–4 节**，缺任意一节视为未完成，`status` 保持 `draft`。
2. **第 1 节有可推导的过程**，不是结论罗列。
3. **第 2 节有具体的执行细节**，不是"这样更快"的定性描述。
4. **第 4 节至少记录一个真实的失败模式**。

第 5、6 节为可选。

### 7.3 篇幅指引

- 单个文档聚焦**一个概念**，不追求大而全。超过约 500 行说明该拆分。
- 宁可拆成两篇并建立 `prerequisites` 关系，也不要写一篇长文。

---

## 8. 命名规范

| 对象 | 规范 | 示例 |
|---|---|---|
| 一级域目录 | `NNN-kebab-case`，三位数，间隔 10 | `20-gpu-architecture` |
| 二级/三级目录 | `kebab-case`，无编号 | `memory-system` |
| 文档文件 | `NN-kebab-case.md`，两位数，域内唯一 | `02-coalescing-and-cache-line.md` |
| 代码文件 | `NN_snake_case.ext`，编号与文档对应 | `02_coalescing_demo.cu` |
| 目录索引 | 固定名 `README.md` | — |
| 临时/草稿 | 前缀下划线 | `_template.md`, `_draft-*.md` |

---

## 9. 首期交付范围

### 9.1 交付物清单

**A. 仓库基础**
- `git init` 并完成首次提交
- `README.md`：知识库定位、16 个一级域的说明、学习路径图、如何使用
- `CONTRIBUTING.md`：命名规范、frontmatter 规范、深度标准、链接约定、新增文档流程
- `_template.md`：可直接复制的文档模板

**B. 完整目录骨架**
- 16 个一级域及第 3.2 节列出的全部二级目录
- 每个一级域与二级主题目录放置 `README.md` 占位，内容包括：本域定位、计划覆盖的主题清单、与其他域的关系

**C. `_meta/` 三个文件**
- `glossary.md`：术语表中英对照，首批收录 GPU 与量化核心术语
- `index.md`：全库索引，按域组织
- `roadmap.md`：待写清单与进度跟踪

**D. 样板链：`20-gpu-architecture/` 六篇文档**

`execution-model/`：

| 文档 | 数学/算法推导 | 执行映射 | 踩坑与失败模式 |
|---|---|---|---|
| `01-sm-internals.md` | 寄存器文件容量与最大常驻 warp 数的约束关系 | SM 内部组成（CUDA core / warp scheduler / register file / LSU / SFU）；warp 为何是调度最小单位 | 跨代 SM 不能直接比数量；寄存器文件是预算而非缓存 |
| `02-simt-and-divergence.md` | 发散导致的**有效并行度**下降模型 | warp 内 32 线程锁步、reconvergence、SASS 上 `BRA/BSSY/BSYNC` 的实际形态、predication | 以为发散只来自 `if/else`；循环边界不齐同样发散 |
| `03-occupancy-and-latency-hiding.md` | 延迟隐藏所需并发度推导（Little's Law）；占用率与寄存器/共享内存的约束方程 | warp 切换如何掩盖访存延迟；指令级并行与线程级并行的分工 | 占用率不是越高越好；高占用反而变慢的成因 |

`memory-system/`：

| 文档 | 数学/算法推导 | 执行映射 | 踩坑与失败模式 |
|---|---|---|---|
| `01-hierarchy-overview.md` | 各级存储容量/带宽/延迟的量级对比；分级的必要性论证 | 片内 vs 片外；寄存器溢出到 local memory 的后果 | 把 shared memory 当通用缓存；忽略 L2 容量导致命中率崩塌 |
| `02-coalescing-and-cache-line.md` | 事务数推导：32 线程 × 4B 连续 = 1 事务；跨步访问 = 32 事务 | cache line 与 L1 sector；合并访存的判定规则 | AoS vs SoA 的真实代价；以为"地址连续"就等于"合并" |
| `03-shared-memory-bank-conflict.md` | `bank = (addr / 4) % 32` 寻址模型；冲突度计算 | bank 结构与多播/broadcast；64-bit 访问下的冲突判定 | padding 方向搞反；以为加 padding 总是有益 |

**E. 样板链配套代码**（自包含可编译）

```
20-gpu-architecture/execution-model/code/
├── 01_divergence_demo.cu
├── 02_occupancy_probe.cu
└── Makefile

20-gpu-architecture/memory-system/code/
├── 01_bandwidth_probe.cu
├── 02_coalescing_demo.cu
├── 03_bank_conflict_demo.cu
└── Makefile
```

代码要求：`nvcc` 可直接编译、有合理的默认参数、输出可直接验证文档中的结论、注释说明观测要点。

### 9.2 不在首期范围

- `20-gpu-architecture/` 的其余三个二级目录（tensor-core / interconnect / architecture-evolution）只放 README 占位
- 其余 15 个一级域只放 README 占位
- 索引生成脚本
- 静态站点

### 9.3 验收标准

1. `knowledge/` 是 git 仓库，工作区干净，有完整的首次提交。
2. 目录结构与第 3.2 节完全一致。
3. 每个一级域都有 README，说明该域定位与计划内容。
4. 6 篇样板文档全部满足第 7.2 节的深度标准，`status` 字段为 `draft`（因为第 5 节实测数据未填）。
5. 6 个 `.cu` 文件均可通过 `nvcc` 编译（若环境有 GPU 则可运行并复现文档结论）。
6. `_meta/glossary.md` 收录不少于 30 条术语。
7. `CONTRIBUTING.md` 中的规范与本文档第 6、7、8 节一致，无矛盾。

---

## 10. 后续演进

以下内容不在首期，但方案已为其预留接口：

- **索引自动化**：文档量增长后，用脚本扫描 frontmatter 生成 `_meta/index.md` 与各域导航。
- **向量化检索**：frontmatter 的 `slug` / `domain` / `tags` / `prerequisites` 已构成检索元数据；单文件单概念的切分已保证 chunk 质量。
- **内容填充**：按 `_meta/roadmap.md` 的顺序，以首期 6 篇为质量标尺逐域填充。

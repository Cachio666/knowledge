# 大模型全栈知识库

从 GPU 硬件底层到推理引擎上层，覆盖训练与推理全链路的学习型知识库。

## 这个知识库是什么

不是教程摘抄，而是**原理推导 + 执行映射 + 失败模式**三件套的沉淀。每篇文档回答三个问题：

1. 这东西的数学/算法原理是什么，怎么推导出来的？
2. 它在真实系统（SM、缓存、dispatch、调度器）上实际发生了什么？
3. 什么情况下会失效、退化、或给出反直觉的结果？

## 目录结构

按技术栈分层，自下而上编号。编号三位数、间隔 10，插入新领域不需重排。

| 域 | 覆盖内容 |
|---|---|
| `00-meta` | 知识库自身：术语表、索引、进度表 |
| `10-foundations` | 浮点与数值、数值线性代数、性能模型（roofline、Amdahl） |
| `20-gpu-architecture` | SM 与执行模型、存储层次、Tensor Core、互联、架构演进 |
| `30-gpu-programming` | CUDA、Triton、PTX/SASS、CUTLASS |
| `40-kernels` | GEMM、attention、reduce、softmax、融合算子、MoE kernel |
| `50-performance` | profiling 工具、瓶颈定位、调优实战 |
| `60-compilers` | MLIR、TVM、Inductor、XLA、自动调优 |
| `70-frameworks` | PyTorch、TensorFlow、JAX |
| `80-model-architectures` | Transformer、注意力变体、MoE、归一化、位置编码 |
| `90-training` | 反向传播、混合精度、优化器、训练稳定性、检查点 |
| `100-parallelism` | DP、TP、PP、SP/CP、EP、ZeRO/FSDP |
| `110-communication` | 集合通信、NCCL、网络拓扑、通信计算重叠 |
| `120-inference` | KV cache、batching 调度、解码策略、推理引擎、部署 |
| `130-quantization` | 数值格式、量化理论、PTQ、QAT、量化算子、KV cache 量化 |
| `140-papers` | 论文精读 |
| `150-playground` | 实验场 |

`00-meta/` 下有三个文件：[术语表](00-meta/glossary.md)、[索引](00-meta/index.md)、[进度表](00-meta/roadmap.md)。

## 学习路径图

自下而上，每一层都是下一层的前提。箭头表示「读懂了左边再读右边」。

```
10-foundations  浮点与数值 / 性能模型
      │
      ▼
20-gpu-architecture   SM 与执行模型 → 存储层次 → Tensor Core → 互联
      │
      ▼
30-gpu-programming    CUDA → Triton → PTX/SASS → CUTLASS
      │
      ▼
40-kernels            GEMM → attention → reduce / softmax / 融合算子
      │
      ├──────────────► 50-performance   瓶颈定位与调优（横切，随时可读）
      │
      ▼
60-compilers          MLIR → TVM → Inductor → XLA
      │
      ▼
70-frameworks         PyTorch / TensorFlow / JAX
      │
      ├──────────────► 80-model-architectures   要优化的对象
      │
      ▼
90-training           反向传播 → 混合精度 → 优化器 → 训练稳定性
      │
      ├──────────────► 100-parallelism   DP / TP / PP / EP / ZeRO
      ├──────────────► 110-communication 集合通信 / NCCL / 拓扑
      │
      ▼
120-inference         KV cache → 调度 → 解码 → 推理引擎
      │
      └──────────────► 130-quantization  量化与压缩（横切训练与推理）

140-papers      论文精读（随时可读）
150-playground  实验场（随时可写）
```

## 怎么用

**按依赖顺序学**：每篇文档头部的 frontmatter 里有 `prerequisites`（必须先读）和 `related`（建议延伸）。从 `prerequisites` 为空的文档开始，跟着链接走。

**按需查找**：`00-meta/index.md` 是全库索引；各域 README 列出该域的主题清单。

**看进度**：`00-meta/roadmap.md` 记录每个主题的完成状态。

## 文档与代码的关系

编号即外键。文档 `NN-<名称>.md` 与代码 `code/NN_<名称>.<扩展名>` 一一对应：

```
20-gpu-architecture/memory-system/
├── 01-hierarchy-overview.md          ←→  code/01_bandwidth_probe.cu
├── 02-coalescing-and-cache-line.md   ←→  code/02_coalescing_demo.cu
├── 03-shared-memory-bank-conflict.md ←→  code/03_bank_conflict_demo.cu
└── code/
    ├── 01_bandwidth_probe.cu
    ├── 02_coalescing_demo.cu
    ├── 03_bank_conflict_demo.cu
    └── Makefile
```

正文内嵌讲解所需的关键代码片段；完整可编译版本在 `code/` 下，自带 `Makefile`，可单独拷贝到有 CUDA 环境的机器运行。

## 贡献与规范

写作前请读 [CONTRIBUTING.md](CONTRIBUTING.md)。新文档可从 [_template.md](_template.md) 复制。

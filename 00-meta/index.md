# 全库索引

按一级域组织。每个域下列出该域已创建的主题目录；尚未创建的主题见 [roadmap.md](roadmap.md)。

> 本文件手工维护。当文档数超过 50 时改为脚本生成（见设计文档 §11）。

## 00-meta
- [术语表](glossary.md)
- [索引](index.md)（本文件）
- [进度表](roadmap.md)

## 10-foundations
- [floating-point](../10-foundations/floating-point/) —— IEEE754 格式 / 舍入与误差模型 / 求和与 FMA / 低精度格式
  - [01-ieee754-format](../10-foundations/floating-point/01-ieee754-format.md)
  - [02-rounding-and-error-model](../10-foundations/floating-point/02-rounding-and-error-model.md)
  - [03-summation-and-accumulation](../10-foundations/floating-point/03-summation-and-accumulation.md)
  - [04-low-precision-formats](../10-foundations/floating-point/04-low-precision-formats.md)
- [numerical-linear-algebra](../10-foundations/numerical-linear-algebra/) —— 范数与条件数 / GEMM 误差 / 分解稳定性 / 迭代法
  - [01-norms-and-condition-number](../10-foundations/numerical-linear-algebra/01-norms-and-condition-number.md)
  - [02-gemm-rounding-error](../10-foundations/numerical-linear-algebra/02-gemm-rounding-error.md)
  - [03-matrix-factorization-stability](../10-foundations/numerical-linear-algebra/03-matrix-factorization-stability.md)
  - [04-iterative-methods](../10-foundations/numerical-linear-algebra/04-iterative-methods.md)
- [performance-models](../10-foundations/performance-models/) —— roofline 模型 / Amdahl 扩展性 / 实测 roofline
  - [01-roofline-model](../10-foundations/performance-models/01-roofline-model.md)
  - [02-amdahl-and-scaling](../10-foundations/performance-models/02-amdahl-and-scaling.md)
  - [03-measuring-roofline](../10-foundations/performance-models/03-measuring-roofline.md)

## 20-gpu-architecture
- [execution-model](../20-gpu-architecture/execution-model/) —— SM 结构 / SIMT 与发散 / 占用率与延迟隐藏
  - [01-sm-internals](../20-gpu-architecture/execution-model/01-sm-internals.md)
  - [02-simt-and-divergence](../20-gpu-architecture/execution-model/02-simt-and-divergence.md)
  - [03-occupancy-and-latency-hiding](../20-gpu-architecture/execution-model/03-occupancy-and-latency-hiding.md)
- [memory-system](../20-gpu-architecture/memory-system/) —— 存储层次 / 访存合并 / 共享内存 bank 冲突
  - [01-hierarchy-overview](../20-gpu-architecture/memory-system/01-hierarchy-overview.md)
  - [02-coalescing-and-cache-line](../20-gpu-architecture/memory-system/02-coalescing-and-cache-line.md)
  - [03-shared-memory-bank-conflict](../20-gpu-architecture/memory-system/03-shared-memory-bank-conflict.md)
- [tensor-core](../20-gpu-architecture/tensor-core/) —— MMA 原语 / fragment 布局 / wgmma / tcgen05 与 TMEM / 结构化稀疏
  - [01-mma-primitive](../20-gpu-architecture/tensor-core/01-mma-primitive.md)
  - [02-fragment-layout](../20-gpu-architecture/tensor-core/02-fragment-layout.md)
  - [03-wgmma-and-pipeline](../20-gpu-architecture/tensor-core/03-wgmma-and-pipeline.md)
  - [04-tcgen05-and-tmem](../20-gpu-architecture/tensor-core/04-tcgen05-and-tmem.md)
  - [05-structured-sparsity](../20-gpu-architecture/tensor-core/05-structured-sparsity.md)
- [interconnect](../20-gpu-architecture/interconnect/) —— 带宽口径 / 拓扑与 bisection / 拓扑受限的集合通信 / 标称与可达
  - [01-bandwidth-conventions](../20-gpu-architecture/interconnect/01-bandwidth-conventions.md)
  - [02-topology-and-bisection](../20-gpu-architecture/interconnect/02-topology-and-bisection.md)
  - [03-collectives-on-topology](../20-gpu-architecture/interconnect/03-collectives-on-topology.md)
  - [04-measuring-bandwidth](../20-gpu-architecture/interconnect/04-measuring-bandwidth.md)
- [architecture-evolution](../20-gpu-architecture/architecture-evolution/) —— 能力断点 / 跨代趋势 / 架构不变量
  - [01-capability-breakpoints](../20-gpu-architecture/architecture-evolution/01-capability-breakpoints.md)
  - [02-scaling-trends](../20-gpu-architecture/architecture-evolution/02-scaling-trends.md)
  - [03-invariants](../20-gpu-architecture/architecture-evolution/03-invariants.md)

## 30-gpu-programming
尚未开始。

## 40-kernels
尚未开始。

## 50-performance
尚未开始。

## 60-compilers
尚未开始。

## 70-frameworks
尚未开始。

## 80-model-architectures
尚未开始。

## 90-training
尚未开始。

## 100-parallelism
尚未开始。

## 110-communication
尚未开始。

## 120-inference
尚未开始。

## 130-quantization
尚未开始。

## 140-papers
尚未开始。

## 150-playground
尚未开始。

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

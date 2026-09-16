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
- [gemm](../40-kernels/gemm/) —— 朴素 → 共享内存分块 → 寄存器分块 → tensor core 与异步流水 → persistent kernel 与 tcgen05 → split-K 与 Stream-K
  - [01-naive-gemm-and-roofline](../40-kernels/gemm/01-naive-gemm-and-roofline.md)
  - [02-shared-memory-tiling](../40-kernels/gemm/02-shared-memory-tiling.md)
  - [03-register-tiling-and-occupancy](../40-kernels/gemm/03-register-tiling-and-occupancy.md)
  - [04-tensor-core-and-async-pipeline](../40-kernels/gemm/04-tensor-core-and-async-pipeline.md)
  - [05-persistent-kernel-and-tcgen05](../40-kernels/gemm/05-persistent-kernel-and-tcgen05.md)
  - [06-split-k-and-stream-k](../40-kernels/gemm/06-split-k-and-stream-k.md)
- [attention](../40-kernels/attention/) —— attention 的计算形态 / flash attention 与在线 softmax / flash attention 反向 / warp specialization 与变体 / 解码阶段 kernel
  - [01-attention-as-a-kernel](../40-kernels/attention/01-attention-as-a-kernel.md)
  - [02-flash-attention](../40-kernels/attention/02-flash-attention.md)
  - [03-flash-attention-backward](../40-kernels/attention/03-flash-attention-backward.md)
  - [04-warp-specialization-and-variants](../40-kernels/attention/04-warp-specialization-and-variants.md)
  - [05-decoding-attention](../40-kernels/attention/05-decoding-attention.md)
- [reduce-scan](../40-kernels/reduce-scan/) —— 树形归约 / warp shuffle 与多级归约 / Hillis-Steele 与 Blelloch 扫描 / 跨 block 与确定性
  - [01-reduction-algorithms](../40-kernels/reduce-scan/01-reduction-algorithms.md)
  - [02-warp-shuffle-and-hierarchy](../40-kernels/reduce-scan/02-warp-shuffle-and-hierarchy.md)
  - [03-scan-algorithms](../40-kernels/reduce-scan/03-scan-algorithms.md)
  - [04-multi-block-and-determinism](../40-kernels/reduce-scan/04-multi-block-and-determinism.md)
- [softmax-normalization](../40-kernels/softmax-normalization/) —— 数值稳定 softmax / 在线 softmax 与重标定 / LayerNorm·RMSNorm 与 Welford
  - [01-softmax-stability](../40-kernels/softmax-normalization/01-softmax-stability.md)
  - [02-online-softmax](../40-kernels/softmax-normalization/02-online-softmax.md)
  - [03-normalization-fusion](../40-kernels/softmax-normalization/03-normalization-fusion.md)
- [elementwise-fusion](../40-kernels/elementwise-fusion/) —— 访存受限算子的上界 / 融合与流量 / 向量化访存与 grid-stride
  - [01-memory-bound-limit](../40-kernels/elementwise-fusion/01-memory-bound-limit.md)
  - [02-fusion-and-traffic](../40-kernels/elementwise-fusion/02-fusion-and-traffic.md)
  - [03-vectorized-and-grid-stride](../40-kernels/elementwise-fusion/03-vectorized-and-grid-stride.md)
- [moe-kernels](../40-kernels/moe-kernels/) —— MoE 的计算形态 / grouped GEMM / 路由与 permute·unpermute
  - [01-moe-compute-pattern](../40-kernels/moe-kernels/01-moe-compute-pattern.md)
  - [02-grouped-gemm](../40-kernels/moe-kernels/02-grouped-gemm.md)
  - [03-permute-and-unpermute](../40-kernels/moe-kernels/03-permute-and-unpermute.md)

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

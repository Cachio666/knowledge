# 实施进度表

> 最后更新：2026-09-20
> 进度：34 / 65 主题完成

状态标记：`- [x]` 完成 · `- [ ] <topic> (进行中)` 进行中 · `- [ ] <topic>` 未开始

计数口径：分母 65 = 有固定二级结构的 14 个域的二级目录总数（`00-meta`、`140-papers`、`150-playground` 不计入）。新增或删除主题时须同步更新分母。

## 10-foundations
- [x] floating-point
- [x] numerical-linear-algebra
- [x] performance-models

## 20-gpu-architecture
- [x] execution-model
- [x] memory-system
- [x] tensor-core
- [x] interconnect
- [x] architecture-evolution

## 30-gpu-programming
- [x] cuda
- [ ] triton (进行中)
- [ ] ptx-sass (进行中)
- [ ] cutlass (进行中)

## 40-kernels
- [x] gemm
- [x] attention
- [x] reduce-scan
- [x] softmax-normalization
- [x] elementwise-fusion
- [x] moe-kernels

## 45-kernel-lifecycle
- [x] requirement-discovery
- [x] operator-design
- [x] implementation
- [x] convergence-and-tuning

## 50-performance
- [ ] profiling-tools
- [ ] bottleneck-diagnosis
- [ ] tuning-practice

## 60-compilers
- [x] mlir
- [x] tvm
- [x] inductor
- [x] xla
- [x] auto-tuning

## 70-frameworks
- [x] pytorch
- [x] tensorflow
- [x] jax

## 80-model-architectures
- [ ] transformer
- [ ] attention-variants
- [ ] moe
- [ ] normalization
- [ ] positional-encoding

## 90-training
- [ ] autograd-and-backward
- [ ] mixed-precision
- [ ] optimizers
- [ ] initialization-and-stability
- [ ] checkpointing

## 100-parallelism
- [ ] data-parallel
- [ ] tensor-parallel
- [ ] pipeline-parallel
- [ ] sequence-context-parallel
- [ ] expert-parallel
- [ ] zero-fsdp

## 110-communication
- [ ] collectives
- [ ] nccl
- [ ] network-topology
- [ ] overlap

## 120-inference
- [ ] kv-cache
- [ ] batching-and-scheduling
- [ ] decoding-strategies
- [ ] engines
- [ ] serving-infra

## 130-quantization
- [x] numerical-formats
- [x] theory
- [x] ptq
- [x] qat
- [x] quantized-kernels
- [x] kv-cache-quantization
- [x] sparsity-and-distillation

## 140-papers
（无固定二级结构，按需追加）

## 150-playground
（无固定二级结构，按需追加）

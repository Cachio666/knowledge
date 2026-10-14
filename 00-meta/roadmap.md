# 实施进度表

> 最后更新：2026-10-13
> 进度：67 / 68 主题完成

状态标记：`- [x]` 完成 · `- [ ] <topic> (进行中)` 进行中 · `- [ ] <topic>` 未开始

计数口径：分母 68 = 有固定二级结构的 15 个域的二级目录总数（`00-meta`、`140-papers`、`150-playground` 不计入）。新增或删除主题时须同步更新分母。

> ⚠ **2026-10-13 的两次调整**（都在 `130`）：
> ① `130-quantization` 重建为 `130-model-optimization`（大模型优化技术）——原 `sparsity-and-distillation`
>    一题拆为 `sparsity` / `pruning` / `distillation` 三题，另增 `overview`、`low-rank`（均已建成）与 `peft`（未开始）；
> ② 量化家族的 6 个子主题（`numerical-formats` / `theory` / `ptq` / `qat` / `quantized-kernels` /
>    `kv-cache-quantization`）**合并为单一 `quantization` 主题**（20 篇文档编号 01–20）。
> ⇒ `130` 的主题数由 7 → 12 → **7**，全库分母由 68 → 73 → **68**（两次调整互相抵消）。
> ★ 合并**不丢内容**：20 篇文档与 20 个编号程序全部保留，只是编号与目录归属变了。

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
- [x] triton
- [x] ptx-sass
- [x] cutlass

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
- [x] profiling-tools
- [x] bottleneck-diagnosis
- [x] tuning-practice

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
- [x] transformer
- [x] attention-variants
- [x] moe
- [x] normalization
- [x] positional-encoding

## 90-training
- [x] autograd-and-backward
- [x] mixed-precision
- [x] optimizers
- [x] initialization-and-stability
- [x] checkpointing

## 100-parallelism
- [x] data-parallel
- [x] tensor-parallel
- [x] pipeline-parallel
- [x] sequence-context-parallel
- [x] expert-parallel
- [x] zero-fsdp

## 110-communication
- [x] collectives
- [x] nccl
- [x] network-topology
- [x] overlap

## 120-inference
- [x] kv-cache
- [x] batching-and-scheduling
- [x] decoding-strategies
- [x] engines
- [x] serving-infra

## 130-model-optimization
- [x] overview
- [x] quantization
- [x] sparsity
- [x] pruning
- [x] distillation
- [x] low-rank
- [ ] peft

## 140-papers
（无固定二级结构，按需追加）

## 150-playground
（无固定二级结构，按需追加）

## 160-model
- [x] glm
- [x] deepseek
- [x] kimi

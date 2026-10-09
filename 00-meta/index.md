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
- [tensor-core](../20-gpu-architecture/tensor-core/) —— MMA 原语与 tile 形状 / fragment 布局 / wgmma 与流水 / tcgen05 与 TMEM / 2:4 结构化稀疏
  - [01-mma-primitive](../20-gpu-architecture/tensor-core/01-mma-primitive.md)
  - [02-fragment-layout](../20-gpu-architecture/tensor-core/02-fragment-layout.md)
  - [03-wgmma-and-pipeline](../20-gpu-architecture/tensor-core/03-wgmma-and-pipeline.md)
  - [04-tcgen05-and-tmem](../20-gpu-architecture/tensor-core/04-tcgen05-and-tmem.md)
  - [05-structured-sparsity](../20-gpu-architecture/tensor-core/05-structured-sparsity.md)
- [interconnect](../20-gpu-architecture/interconnect/) —— 带宽口径 / 拓扑与 bisection / 拓扑受限的集合通信 / 标称与可达的差距
  - [01-bandwidth-conventions](../20-gpu-architecture/interconnect/01-bandwidth-conventions.md)
  - [02-topology-and-bisection](../20-gpu-architecture/interconnect/02-topology-and-bisection.md)
  - [03-collectives-on-topology](../20-gpu-architecture/interconnect/03-collectives-on-topology.md)
  - [04-measuring-bandwidth](../20-gpu-architecture/interconnect/04-measuring-bandwidth.md)
- [architecture-evolution](../20-gpu-architecture/architecture-evolution/) —— 能力断点 / 跨代趋势 / 架构不变量
  - [01-capability-breakpoints](../20-gpu-architecture/architecture-evolution/01-capability-breakpoints.md)
  - [02-scaling-trends](../20-gpu-architecture/architecture-evolution/02-scaling-trends.md)
  - [03-invariants](../20-gpu-architecture/architecture-evolution/03-invariants.md)

## 30-gpu-programming


- [cuda](../30-gpu-programming/cuda/) —— 线程层次与同步契约 / 流与事件的依赖 DAG / 内存管理与 CUDA Graph
  - [01-thread-hierarchy-and-sync](../30-gpu-programming/cuda/01-thread-hierarchy-and-sync.md)
  - [02-streams-and-events](../30-gpu-programming/cuda/02-streams-and-events.md)
  - [03-memory-and-graphs](../30-gpu-programming/cuda/03-memory-and-graphs.md)
- [triton](../30-gpu-programming/triton/) —— 块级程序模型 / 边界掩码 / layout 泄漏 / 归约的跨 warp 代价
  - [01-block-program-model](../30-gpu-programming/triton/01-block-program-model.md)
  - [02-masking-and-boundaries](../30-gpu-programming/triton/02-masking-and-boundaries.md)
  - [03-layout-and-leaks](../30-gpu-programming/triton/03-layout-and-leaks.md)
  - [04-reduction-cost](../30-gpu-programming/triton/04-reduction-cost.md)
- [ptx-sass](../30-gpu-programming/ptx-sass/) —— PTX 作为虚拟 ISA / warp 级执行 / 访存计数与 PTX→SASS 落差
  - [01-ptx-as-virtual-isa](../30-gpu-programming/ptx-sass/01-ptx-as-virtual-isa.md)
  - [02-warp-level-execution](../30-gpu-programming/ptx-sass/02-warp-level-execution.md)
  - [03-memory-and-sass-gap](../30-gpu-programming/ptx-sass/03-memory-and-sass-gap.md)
- [cutlass](../30-gpu-programming/cutlass/) —— CuTe layout 代数 / 层次化 tiling / swizzle 与 collective
  - [01-layout-algebra](../30-gpu-programming/cutlass/01-layout-algebra.md)
  - [02-hierarchical-tiling](../30-gpu-programming/cutlass/02-hierarchical-tiling.md)
  - [03-swizzle-and-collective](../30-gpu-programming/cutlass/03-swizzle-and-collective.md)

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

## 45-kernel-lifecycle
- [requirement-discovery](../45-kernel-lifecycle/requirement-discovery/) —— 从 config 推算子清单 / roofline 判定 / 优化价值排序
  - [01-model-to-operator-inventory](../45-kernel-lifecycle/requirement-discovery/01-model-to-operator-inventory.md)
  - [02-roofline-verdict](../45-kernel-lifecycle/requirement-discovery/02-roofline-verdict.md)
  - [03-optimization-priority](../45-kernel-lifecycle/requirement-discovery/03-optimization-priority.md)
- [operator-design](../45-kernel-lifecycle/operator-design/) —— 接口契约 / 数值设计 / 分块与边界
  - [01-interface-contract](../45-kernel-lifecycle/operator-design/01-interface-contract.md)
  - [02-numerical-design](../45-kernel-lifecycle/operator-design/02-numerical-design.md)
  - [03-tiling-and-boundaries](../45-kernel-lifecycle/operator-design/03-tiling-and-boundaries.md)
- [implementation](../45-kernel-lifecycle/implementation/) —— 参考实现先行 / 验证矩阵 / 调试顺序
  - [01-reference-first](../45-kernel-lifecycle/implementation/01-reference-first.md)
  - [02-verification-matrix](../45-kernel-lifecycle/implementation/02-verification-matrix.md)
  - [03-debug-order](../45-kernel-lifecycle/implementation/03-debug-order.md)
- [convergence-and-tuning](../45-kernel-lifecycle/convergence-and-tuning/) —— 基准测量卫生 / 显著性判定 / 收敛与停止
  - [01-benchmark-hygiene](../45-kernel-lifecycle/convergence-and-tuning/01-benchmark-hygiene.md)
  - [02-significance](../45-kernel-lifecycle/convergence-and-tuning/02-significance.md)
  - [03-convergence](../45-kernel-lifecycle/convergence-and-tuning/03-convergence.md)

## 50-performance
- [profiling-tools](../50-performance/profiling-tools/) —— 工具作为仪器：三种采集机制 / replay 与多 pass / 默认值不中立 / 归因
  - [01-collection-mechanisms](../50-performance/profiling-tools/01-collection-mechanisms.md)
  - [02-replay-and-multipass](../50-performance/profiling-tools/02-replay-and-multipass.md)
  - [03-defaults-and-attribution](../50-performance/profiling-tools/03-defaults-and-attribution.md)
- [bottleneck-diagnosis](../50-performance/bottleneck-diagnosis/) —— 从读数到判据：延迟 vs 带宽 / 读计数器的反模式 / 一次完整诊断
  - [01-latency-and-bandwidth](../50-performance/bottleneck-diagnosis/01-latency-and-bandwidth.md)
  - [02-reading-counters](../50-performance/bottleneck-diagnosis/02-reading-counters.md)
  - [03-diagnosis-walkthrough](../50-performance/bottleneck-diagnosis/03-diagnosis-walkthrough.md)
- [tuning-practice](../50-performance/tuning-practice/) —— σ 与 τ 的标定 / 真实测量噪声下的搜索 / A-B 的实验编排
  - [01-measuring-sigma-and-tau](../50-performance/tuning-practice/01-measuring-sigma-and-tau.md)
  - [02-noise-and-search](../50-performance/tuning-practice/02-noise-and-search.md)
  - [03-ab-orchestration](../50-performance/tuning-practice/03-ab-orchestration.md)

## 60-compilers
- [mlir](../60-compilers/mlir/) —— IR 结构与方言 / tensor 与 memref / 渐进式 lowering / 编写 pass
  - [01-ir-and-dialects](../60-compilers/mlir/01-ir-and-dialects.md)
  - [02-tensor-vs-memref](../60-compilers/mlir/02-tensor-vs-memref.md)
  - [03-progressive-lowering](../60-compilers/mlir/03-progressive-lowering.md)
  - [04-writing-a-pass](../60-compilers/mlir/04-writing-a-pass.md)
- [tvm](../60-compilers/tvm/) —— 计算与调度分离 / 调度原语与物化位置 / 两条 TIR 轨道与 tile 原语 / 自动调度
  - [01-compute-and-schedule](../60-compilers/tvm/01-compute-and-schedule.md)
  - [02-schedule-primitives](../60-compilers/tvm/02-schedule-primitives.md)
  - [03-native-track-and-tile-primitives](../60-compilers/tvm/03-native-track-and-tile-primitives.md)
  - [04-auto-scheduling](../60-compilers/tvm/04-auto-scheduling.md)
- [inductor](../60-compilers/inductor/) —— 图捕获与 graph break / 融合与调度 / codegen 与后端 / 动态形状与重编译
  - [01-graph-capture](../60-compilers/inductor/01-graph-capture.md)
  - [02-fusion-scheduling](../60-compilers/inductor/02-fusion-scheduling.md)
  - [03-codegen](../60-compilers/inductor/03-codegen.md)
  - [04-dynamic-shapes](../60-compilers/inductor/04-dynamic-shapes.md)
- [xla](../60-compilers/xla/) —— HLO 与形状推断 / 融合 kind 与三层判据 / 布局指派与内存空间
  - [01-hlo-and-shape-inference](../60-compilers/xla/01-hlo-and-shape-inference.md)
  - [02-fusion](../60-compilers/xla/02-fusion.md)
  - [03-layout-and-memory](../60-compilers/xla/03-layout-and-memory.md)
- [auto-tuning](../60-compilers/auto-tuning/) —— 搜索空间的名义基数与有效基数 / 代价模型与排序能力 / 搜索策略与预算
  - [01-search-space](../60-compilers/auto-tuning/01-search-space.md)
  - [02-cost-models](../60-compilers/auto-tuning/02-cost-models.md)
  - [03-search-algorithms](../60-compilers/auto-tuning/03-search-algorithms.md)

## 70-frameworks
- [pytorch](../70-frameworks/pytorch/) —— dispatch 与 aten / autograd / 张量内存模型 / 自定义算子与编译入口
  - [01-dispatch-and-aten](../70-frameworks/pytorch/01-dispatch-and-aten.md)
  - [02-autograd](../70-frameworks/pytorch/02-autograd.md)
  - [03-tensor-memory-model](../70-frameworks/pytorch/03-tensor-memory-model.md)
  - [04-custom-operators](../70-frameworks/pytorch/04-custom-operators.md)
- [tensorflow](../70-frameworks/tensorflow/) —— graph 与 eager（TraceType） / 控制流与 XLA / 自定义算子与导出
  - [01-graph-and-eager](../70-frameworks/tensorflow/01-graph-and-eager.md)
  - [02-control-flow-and-xla](../70-frameworks/tensorflow/02-control-flow-and-xla.md)
  - [03-custom-ops-and-export](../70-frameworks/tensorflow/03-custom-ops-and-export.md)
- [jax](../70-frameworks/jax/) —— jit 与追踪（cache key） / 变换（vmap 与 jaxpr） / 分片与心智差异
  - [01-jit-and-tracing](../70-frameworks/jax/01-jit-and-tracing.md)
  - [02-transformations](../70-frameworks/jax/02-transformations.md)
  - [03-sharding-and-differences](../70-frameworks/jax/03-sharding-and-differences.md)

## 80-model-architectures
- [transformer](../80-model-architectures/transformer/) —— 一个 block 的构成与残差流 / FFN 的形态与 `d_ff` 怎么定 / 深度与宽度
  - [01-block-and-residual-stream](../80-model-architectures/transformer/01-block-and-residual-stream.md)
  - [02-ffn-shapes](../80-model-architectures/transformer/02-ffn-shapes.md)
  - [03-depth-and-width](../80-model-architectures/transformer/03-depth-and-width.md)
- [attention-variants](../80-model-architectures/attention-variants/) —— MHA / MQA / GQA 与群组大小 `r` / MLA 的低秩与矩阵吸收 / 滑窗与叠加
  - [01-mha-mqa-gqa](../80-model-architectures/attention-variants/01-mha-mqa-gqa.md)
  - [02-mla-low-rank](../80-model-architectures/attention-variants/02-mla-low-rank.md)
  - [03-window-and-hybrid](../80-model-architectures/attention-variants/03-window-and-hybrid.md)
- [moe](../80-model-architectures/moe/) —— gating 与 routing（token choice vs expert choice）/ 容量因子与两条溢出协议 / 均衡与粒度
  - [01-gating-and-routing](../80-model-architectures/moe/01-gating-and-routing.md)
  - [02-capacity-factor](../80-model-architectures/moe/02-capacity-factor.md)
  - [03-balance-and-granularity](../80-model-architectures/moe/03-balance-and-granularity.md)
- [normalization](../80-model-architectures/normalization/) —— RMSNorm 与 LayerNorm 的不变性 / 放在哪（pre / post / final）与 QK-norm
  - [01-rmsnorm-vs-layernorm](../80-model-architectures/normalization/01-rmsnorm-vs-layernorm.md)
  - [02-placement-and-qk-norm](../80-model-architectures/normalization/02-placement-and-qk-norm.md)
- [positional-encoding](../80-model-architectures/positional-encoding/) —— RoPE 的定义与两套实现约定 / `θ_i` 的谱、混叠与分辨窗口 / 外推方法
  - [01-rope-definition](../80-model-architectures/positional-encoding/01-rope-definition.md)
  - [02-spectrum-and-aliasing](../80-model-architectures/positional-encoding/02-spectrum-and-aliasing.md)
  - [03-extrapolation](../80-model-architectures/positional-encoding/03-extrapolation.md)

## 90-training
- [autograd-and-backward](../90-training/autograd-and-backward/) —— 反向传播与数值梯度对拍 / 重算换显存 / 训练循环里的梯度（累积、清零、裁剪）
  - [01-backward-from-scratch](../90-training/autograd-and-backward/01-backward-from-scratch.md)
  - [02-gradient-checkpointing](../90-training/autograd-and-backward/02-gradient-checkpointing.md)
  - [03-gradient-lifecycle](../90-training/autograd-and-backward/03-gradient-lifecycle.md)
- [mixed-precision](../90-training/mixed-precision/) —— 类型选择表（主权重 / 梯度 / 优化器状态 / loss）/ AMP 与 unscale 的时机 / fp8 训练循环
  - [01-type-selection](../90-training/mixed-precision/01-type-selection.md)
  - [02-amp-execution](../90-training/mixed-precision/02-amp-execution.md)
  - [03-fp8-training](../90-training/mixed-precision/03-fp8-training.md)
- [checkpointing](../90-training/checkpointing/) —— 状态的组成与字节账 / 分片保存与恢复 / 一致性恢复（RNG、dataloader、优化器状态）
  - [01-state-and-bytes](../90-training/checkpointing/01-state-and-bytes.md)
  - [02-sharded-save-and-load](../90-training/checkpointing/02-sharded-save-and-load.md)
  - [03-consistency-and-fault-tolerance](../90-training/checkpointing/03-consistency-and-fault-tolerance.md)
- [optimizers](../90-training/optimizers/) —— SGD 与动量（更新式与状态）/ Adam 与 AdamW（weight decay 的两种语义）/ 学习率调度与梯度裁剪
  - [01-sgd-and-momentum](../90-training/optimizers/01-sgd-and-momentum.md)
  - [02-adam-and-adamw](../90-training/optimizers/02-adam-and-adamw.md)
  - [03-schedules-and-clipping](../90-training/optimizers/03-schedules-and-clipping.md)
- [initialization-and-stability](../90-training/initialization-and-stability/) —— He / Xavier / μP 初始化 / 裁剪与范数监控 / loss spike（现象与归因）
  - [01-initialization](../90-training/initialization-and-stability/01-initialization.md)
  - [02-clipping-and-monitoring](../90-training/initialization-and-stability/02-clipping-and-monitoring.md)
  - [03-instability-and-spikes](../90-training/initialization-and-stability/03-instability-and-spikes.md)

## 100-parallelism
- [data-parallel](../100-parallelism/data-parallel/) —— 梯度为什么是平均不是求和 / 全局 batch 的等价性 / 副本一致性与 DP 的显存瓶颈
  - [01-gradient-averaging](../100-parallelism/data-parallel/01-gradient-averaging.md)
  - [02-gradient-synchronization](../100-parallelism/data-parallel/02-gradient-synchronization.md)
  - [03-redundancy-and-limits](../100-parallelism/data-parallel/03-redundancy-and-limits.md)
- [tensor-parallel](../100-parallelism/tensor-parallel/) —— 列 / 行并行与它们的配对 / 头切分与 vocab 并行 / 通信量为什么也饱和
  - [01-column-and-row](../100-parallelism/tensor-parallel/01-column-and-row.md)
  - [02-heads-and-vocab](../100-parallelism/tensor-parallel/02-heads-and-vocab.md)
  - [03-saturation-and-limits](../100-parallelism/tensor-parallel/03-saturation-and-limits.md)
- [pipeline-parallel](../100-parallelism/pipeline-parallel/) —— 微批与气泡率 / GPipe · 1F1B · interleaved 三种调度 / 各 stage 的负载均衡
  - [01-microbatches-and-bubbles](../100-parallelism/pipeline-parallel/01-microbatches-and-bubbles.md)
  - [02-schedules](../100-parallelism/pipeline-parallel/02-schedules.md)
  - [03-memory-balance](../100-parallelism/pipeline-parallel/03-memory-balance.md)
- [sequence-context-parallel](../100-parallelism/sequence-context-parallel/) —— SP 与 CP 的分工（切的范围不同）/ 跨设备的块统计量重缩放 / 顺序与通信的约束
  - [01-activation-sharding](../100-parallelism/sequence-context-parallel/01-activation-sharding.md)
  - [02-ring-attention](../100-parallelism/sequence-context-parallel/02-ring-attention.md)
  - [03-constraints](../100-parallelism/sequence-context-parallel/03-constraints.md)
- [expert-parallel](../100-parallelism/expert-parallel/) —— 专家分片与两次 all-to-all / 负载不均与 straggler / EP 与其他轴的组合
  - [01-expert-sharding](../100-parallelism/expert-parallel/01-expert-sharding.md)
  - [02-load-imbalance](../100-parallelism/expert-parallel/02-load-imbalance.md)
  - [03-ep-and-other-axes](../100-parallelism/expert-parallel/03-ep-and-other-axes.md)
- [zero-fsdp](../100-parallelism/zero-fsdp/) —— ZeRO 三个阶段的账（分片不是除法）/ 通信量 / FSDP 的策略与分片的正确性
  - [01-three-stages](../100-parallelism/zero-fsdp/01-three-stages.md)
  - [02-communication](../100-parallelism/zero-fsdp/02-communication.md)
  - [03-fsdp-and-correctness](../100-parallelism/zero-fsdp/03-fsdp-and-correctness.md)

## 110-communication
- [collectives](../110-communication/collectives/) —— 集合通信的算法族 / 两段式的阶段划分 / all-to-all 与 P2P 原语
  - [01-algorithm-family](../110-communication/collectives/01-algorithm-family.md)
  - [02-stages-and-costs](../110-communication/collectives/02-stages-and-costs.md)
  - [03-all-to-all-and-p2p](../110-communication/collectives/03-all-to-all-and-p2p.md)
- [nccl](../110-communication/nccl/) —— NCCL 的算法与协议取值 / 拓扑感知 / 报错归因（★ 全靠 [B] 级）
  - [01-architecture-and-selection](../110-communication/nccl/01-architecture-and-selection.md)
  - [02-topology-awareness](../110-communication/nccl/02-topology-awareness.md)
  - [03-failures-and-limits](../110-communication/nccl/03-failures-and-limits.md)
- [network-topology](../110-communication/network-topology/) —— 胖树与 Clos 的参数化 / rail-optimized / RoCE 与 PFC·ECN
  - [01-fat-tree-and-clos](../110-communication/network-topology/01-fat-tree-and-clos.md)
  - [02-rail-optimized](../110-communication/network-topology/02-rail-optimized.md)
  - [03-roce-and-infiniband](../110-communication/network-topology/03-roce-and-infiniband.md)
- [overlap](../110-communication/overlap/) —— 双缓冲与通信分块 / 「能藏多少」的判据 / 异步落盘
  - [01-mechanisms](../110-communication/overlap/01-mechanisms.md)
  - [02-how-much-can-hide](../110-communication/overlap/02-how-much-can-hide.md)
  - [03-async-io](../110-communication/overlap/03-async-io.md)

## 120-inference
尚未开始。

## 130-quantization

- [numerical-formats](../130-quantization/numerical-formats/) —— fp16 / bf16 / tf32 / fp8 / int8 / int4 / nvfp4
  - [01-fp8-family](../130-quantization/numerical-formats/01-fp8-family.md)
  - [02-e8m0-and-mx](../130-quantization/numerical-formats/02-e8m0-and-mx.md)
  - [03-bf16-fp16-tf32](../130-quantization/numerical-formats/03-bf16-fp16-tf32.md)
  - [04-integer-formats](../130-quantization/numerical-formats/04-integer-formats.md)
- [theory](../130-quantization/theory/) —— 量化误差分析 / scale 与 zero-point / outlier 问题
  - [01-error-bounds](../130-quantization/theory/01-error-bounds.md)
  - [02-granularity](../130-quantization/theory/02-granularity.md)
  - [03-outliers-and-propagation](../130-quantization/theory/03-outliers-and-propagation.md)
- [ptq](../130-quantization/ptq/) —— 训练后量化 / 校准方法 / GPTQ / AWQ / SmoothQuant
  - [01-calibration](../130-quantization/ptq/01-calibration.md)
  - [02-gptq](../130-quantization/ptq/02-gptq.md)
  - [03-awq-and-smoothquant](../130-quantization/ptq/03-awq-and-smoothquant.md)
  - [04-rounding](../130-quantization/ptq/04-rounding.md)
- [qat](../130-quantization/qat/) —— 量化感知训练 / 伪量化 / 梯度近似
  - [01-ste](../130-quantization/qat/01-ste.md)
  - [02-fake-quant-and-lsq](../130-quantization/qat/02-fake-quant-and-lsq.md)
  - [03-pact-and-comparison](../130-quantization/qat/03-pact-and-comparison.md)
- [quantized-kernels](../130-quantization/quantized-kernels/) —— 量化算子实现（正文在本域，回链 40-kernels）
  - [01-w8a8](../130-quantization/quantized-kernels/01-w8a8.md)
  - [02-w4a16-packing](../130-quantization/quantized-kernels/02-w4a16-packing.md)
  - [03-kernel-design](../130-quantization/quantized-kernels/03-kernel-design.md)
- [kv-cache-quantization](../130-quantization/kv-cache-quantization/) —— KV cache 量化（正文在本域，回链 120-inference）
  - [01-kv-cache-basics](../130-quantization/kv-cache-quantization/01-kv-cache-basics.md)
  - [02-kivi-asymmetry](../130-quantization/kv-cache-quantization/02-kivi-asymmetry.md)
  - [03-low-bit-and-engines](../130-quantization/kv-cache-quantization/03-low-bit-and-engines.md)
- [sparsity-and-distillation](../130-quantization/sparsity-and-distillation/) —— 结构化与非结构化稀疏 / 知识蒸馏
  - [01-sparsity-granularity](../130-quantization/sparsity-and-distillation/01-sparsity-granularity.md)
  - [02-pruning-criteria](../130-quantization/sparsity-and-distillation/02-pruning-criteria.md)
  - [03-distillation](../130-quantization/sparsity-and-distillation/03-distillation.md)

## 140-papers
尚未开始。

## 150-playground
尚未开始。
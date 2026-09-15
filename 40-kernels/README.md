# 40-kernels 算子内核工程

把原理落成可跑的 kernel，并让它跑到硬件允许的上界。

## 本域覆盖什么

| 主题 | 内容 |
|---|---|
| gemm | 朴素 → tiling → 寄存器分块 → tcgen05 → persistent kernel |
| attention | flash attention 系列演进 / 变体 / 解码阶段 kernel |
| reduce-scan | 归约与扫描的并行算法 / warp shuffle / 多级归约 |
| softmax-normalization | softmax 数值稳定实现 / 在线 softmax / norm 融合 |
| elementwise-fusion | 逐元素算子融合 / 访存受限算子的优化边界 |
| moe-kernels | grouped GEMM / all-to-all / 专家调度 kernel |

## 与其他域的关系

- **上游**：`20-gpu-architecture`（硬件行为）、`30-gpu-programming`（编程工具）、`10-foundations`（roofline 与数值）。
- **下游**：`60-compilers`（编译器要生成的正是这些 kernel）、`120-inference`（引擎的算力上限由这些 kernel 决定）。
- **横切**：量化算子实现（正文在 `130-quantization`，此处回链）。

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)

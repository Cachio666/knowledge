# 10-foundations 数学与数值基础

大模型系统里反复用到、且不理解就会算错的数学与数值前提。

## 本域覆盖什么

| 主题 | 内容 |
|---|---|
| floating-point | IEEE754 格式、舍入模式、误差累积、次正规数 |
| numerical-linear-algebra | 矩阵计算、条件数、数值稳定性、迭代求解 |
| performance-models | roofline 模型、算术强度、Amdahl 定律、延迟-带宽模型 |

## 与其他域的关系

- **上游**：无。本域是全库最底层，`prerequisites` 为空。
- **下游**：`20-gpu-architecture`（延迟-带宽模型）、`40-kernels`（roofline 判断算子性质）、`90-training`（混合精度与数值稳定性）、`130-quantization`（误差分析）。
- **横切**：`performance-models` 中的 roofline 模型（正文在本域，回链 `40-kernels`）。

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)

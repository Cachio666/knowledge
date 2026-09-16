# 60-compilers 编译器与图优化

从计算图到 kernel 的自动化路径。理解它才能在自动生成的效果不好时知道该动哪里。

## 本域覆盖什么

| 主题 | 内容 | 状态 |
|---|---|---|
| [mlir](mlir/) | IR 结构与方言 / tensor 与 memref / 渐进式 lowering / 编写 pass | 已完成 |
| [tvm](tvm/) | 计算与调度分离 / 调度原语与物化位置 / 两条 TIR 轨道与 tile 原语 / 自动调度 | 已完成 |
| [inductor](inductor/) | 图捕获与 graph break / 融合与调度 / codegen 与后端 / 动态形状与重编译 | 已完成 |
| [xla](xla/) | TensorFlow 与 JAX 的图编译 / HLO 与形状推断 / 融合 kind 与三层判据 / 布局指派与内存空间 | 已完成 |
| [auto-tuning](auto-tuning/) | 搜索空间的名义基数与有效基数 / 代价模型与排序能力 / 搜索策略与预算 | 已完成 |

## 与其他域的关系

- **上游**：`70-frameworks`（编译器是框架的一部分）、`30-gpu-programming`（生成的代码是 Triton 或 CUDA）。
- **下游**：`120-inference`（引擎大量依赖编译期优化）。
- **横切**：`inductor`（正文在本域，回链 `70-frameworks/pytorch`）。

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)

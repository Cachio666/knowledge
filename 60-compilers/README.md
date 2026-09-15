# 60-compilers 编译器与图优化

从计算图到 kernel 的自动化路径。理解它才能在自动生成的效果不好时知道该动哪里。

## 本域覆盖什么

| 主题 | 内容 |
|---|---|
| mlir | MLIR 方言体系 / lowering 流程 / pass 编写 |
| tvm | TVM 与 TileLang / 调度原语 / 自动代码生成 |
| inductor | torch.compile 后端 / 图捕获 / codegen / 与 Triton 的关系 |
| xla | TensorFlow 与 JAX 的图编译 / HLO / 算子融合 |
| auto-tuning | 自动调优方法 / 搜索空间设计 / 代价模型 |

## 与其他域的关系

- **上游**：`70-frameworks`（编译器是框架的一部分）、`30-gpu-programming`（生成的代码是 Triton 或 CUDA）。
- **下游**：`120-inference`（引擎大量依赖编译期优化）。
- **横切**：`inductor`（正文在本域，回链 `70-frameworks/pytorch`）。

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)

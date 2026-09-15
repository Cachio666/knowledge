# 70-frameworks 深度学习框架

写模型的地方，也是理解"一个算子从 Python 调用到 GPU 执行经过了多少层"的地方。

## 本域覆盖什么

| 主题 | 内容 |
|---|---|
| pytorch | autograd / dispatch 机制 / aten / 自定义算子 / torch.compile |
| tensorflow | graph 与 eager / XLA 集成 / custom op / SavedModel |
| jax | jit / vmap / pmap / sharding / 与 PyTorch 的心智差异 |

## 与其他域的关系

- **上游**：`30-gpu-programming`（自定义算子要写 CUDA）。
- **下游**：`80-model-architectures`（模型用框架写）、`90-training`、`100-parallelism`、`120-inference`。
- **横切**：torch.compile 与 XLA 的编译器实现（正文在 `60-compilers`，此处回链）。

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)

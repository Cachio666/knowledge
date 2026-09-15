# 90-training 训练系统

把模型训起来，并让它不崩、不慢、可复现。

## 本域覆盖什么

| 主题 | 内容 |
|---|---|
| autograd-and-backward | 计算图构造 / 反向传播实现 / 梯度检查点 |
| mixed-precision | AMP / loss scaling / fp8 训练 / 精度对齐与复现 |
| optimizers | SGD / Adam / AdamW / Muon / 分布式优化器状态 |
| initialization-and-stability | 参数初始化 / 梯度裁剪 / loss spike 归因 |
| checkpointing | 分片保存与恢复 / 容错 / 异步落盘 |

## 与其他域的关系

- **上游**：`70-frameworks`、`10-foundations`（数值稳定性）。
- **下游**：`100-parallelism`（并行策略服务于训练）、`110-communication`（分布式训练的通信开销）。
- **横切**：优化器**未**独立成一级域，因为与训练循环、混合精度、分布式优化器状态强耦合。

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)

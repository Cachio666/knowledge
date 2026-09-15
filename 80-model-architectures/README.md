# 80-model-architectures 模型架构

要优化的对象。所有算子、并行策略、推理优化的收益最终都要落在这个结构上。

## 本域覆盖什么

| 主题 | 内容 |
|---|---|
| transformer | 原版结构 → 现代变体 / 残差设计 / 深度与宽度缩放 |
| attention-variants | MHA / MQA / GQA / MLA / 线性注意力 / 混合注意力 |
| moe | gating 机制 / 专家路由 / 负载均衡 / 容量因子 |
| normalization | LayerNorm / RMSNorm / QK-norm / pre-post 位置之争 |
| positional-encoding | RoPE / ALiBi / 相对位置 / 长度外推方法 |

## 与其他域的关系

- **上游**：`70-frameworks`（模型用框架实现）。
- **下游**：`40-kernels`（attention 变体直接决定 kernel 设计）、`100-parallelism`（MoE 引出专家并行）、`120-inference`（KV cache 的大小由架构决定）。
- **横切**：注意力变体的 kernel 实现在 `40-kernels/attention`，本域只讲架构层面的取舍。

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)

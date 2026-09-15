# 100-parallelism 并行策略

把模型切到多卡上。每种切法都有通信量与显存占用的代价，本域讲清楚这些账怎么算。

## 本域覆盖什么

| 主题 | 内容 |
|---|---|
| data-parallel | DP / DDP / 梯度规约（正文在 110，此处回链） |
| tensor-parallel | TP / 列行并行 / 通信量推导 / 与 TP 组划分 |
| pipeline-parallel | PP / 1F1B / 气泡率分析 / 显存均衡 |
| sequence-context-parallel | SP / CP / ring attention / 长序列切分 |
| expert-parallel | EP / all-to-all 通信模式 |
| zero-fsdp | ZeRO 三阶段 / FSDP / 显存占用核算 |

## 与其他域的关系

- **上游**：`90-training`、`110-communication`（并行策略的通信开销由通信域定性）。
- **下游**：`120-inference`（推理的并行形态与训练不同，但共享概念）。
- **横切**：梯度规约算法（ring allreduce 等）（正文在 `110-communication`，此处回链）。

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)

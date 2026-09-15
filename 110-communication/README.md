# 110-communication 通信与网络

分布式系统的物理约束在这里。通信算法的复杂度与网络拓扑决定了并行的可扩展性上界。

## 本域覆盖什么

| 主题 | 内容 |
|---|---|
| collectives | allreduce / allgather / reducescatter 算法与复杂度 |
| nccl | NCCL 架构 / 调优 / 拓扑感知 / 常见报错归因 |
| network-topology | InfiniBand / RoCE / NVLink 域 / 胖树拓扑 |
| overlap | 通信-计算重叠 / 双缓冲 / 分块流水 |

## 与其他域的关系

- **上游**：`20-gpu-architecture/interconnect`（NVLink 的物理带宽与拓扑）。
- **下游**：`100-parallelism`（并行策略的可行性由通信决定）。
- **横切**：NVLink 与 NVSwitch 的硬件细节（正文在 `20-gpu-architecture/interconnect`，此处回链）。

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)

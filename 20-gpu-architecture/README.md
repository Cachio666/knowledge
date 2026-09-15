# 20-gpu-architecture GPU 体系结构

硬件的静态结构。不理解这一层，后面所有「为什么这个写法快」都没有依据。

## 本域覆盖什么

| 主题 | 内容 | 状态 |
|---|---|---|
| [execution-model](execution-model/) | SM 内部结构 / warp 与 SIMT / 分支发散 / 占用率与延迟隐藏 | 未开始 |
| [memory-system](memory-system/) | 存储层次 / 访存合并 / 共享内存 bank 冲突 | 未开始 |
| tensor-core | MMA 指令族 / wmma / wgmma / tcgen05 / 稀疏 MMA | 未开始 |
| interconnect | NVLink / NVSwitch / PCIe / 多卡拓扑与带宽 | 未开始 |
| architecture-evolution | 跨代横向对比与规律提炼。**规格明细归各机制主题**（如逐代 SM 规格在 `execution-model`），本主题只做横向比较与趋势归纳，不重复罗列规格 | 未开始 |

## 与其他域的关系

- **上游**：`10-foundations`（延迟-带宽模型是理解存储层次的工具）。
- **下游**：`30-gpu-programming`、`40-kernels`、`50-performance`、`110-communication`。
- **横切**：NVLink 带宽与拓扑（正文在本域，回链 `110-communication`）。

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)

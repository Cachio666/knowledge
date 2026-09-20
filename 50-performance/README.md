# 50-performance 性能工程

跨层的性能方法论。本域不重复讲具体算子的优化手法，只讲**怎么定位瓶颈**与**怎么测量**。

## 本域覆盖什么

| 主题 | 内容 |
|---|---|
| profiling-tools | nsight compute / nsight systems / CUPTI / torch profiler |
| bottleneck-diagnosis | 瓶颈定位方法论 / 常见反模式 / 计算-访存-延迟三分法 |
| tuning-practice | 调优实战 / 参数搜索 / 可复现的基准测试方法 |

> ★ **本域的 README 存在，但主题目录一个都没有**——上表三个主题尚无目录，
> 引用它们时按总设计 §4 的**前向声明**处理。

## 与其他域的关系

- **上游**：`20-gpu-architecture`（硬件计数器）、`10-foundations`（roofline 作为判据）。
- **下游**：`40-kernels`、`60-compilers`、`120-inference` 的调优都引用本域的方法论。
- **横切**：本域是横切域。任何文档第 5 节「实测数据」的测量方法都链接到此处，不在各文档里重复描述测量步骤。
- **与 `30-gpu-programming` 的关系**：瓶颈定位常需下钻到 SASS；**读 SASS 的方法与 ISA 语义**在 `30-gpu-programming/ptx-sass`（正文在那边，此处回链），本域不重复 ISA 语义、也不重复语言层的承诺与泄漏。

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)

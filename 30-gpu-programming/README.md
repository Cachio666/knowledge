# 30-gpu-programming GPU 编程模型

从 CUDA C++ 到 Triton、从源码到 SASS，指挥 GPU 的几套语言与抽象层次。

## 本域覆盖什么

| 主题 | 内容 |
|---|---|
| cuda | 线程组织 / 同步原语 / stream / 内存管理 / CUDA Graph |
| triton | Triton 语言模型 / 编译流程 / 常见 kernel 模式 |
| ptx-sass | PTX 与 SASS 指令集 / 寄存器分配 / 指令调度 / 反汇编方法 |
| cutlass | CUTLASS 模板抽象 / 层次化 tiling / collective 设计 |

## 与其他域的关系

- **上游**：`20-gpu-architecture`。不理解 SM 与存储层次，写不出好 kernel。
- **下游**：`40-kernels`（用这些工具写算子）、`60-compilers`（Inductor 生成 Triton 代码，需要读懂才能调优）。
- **横切**：`ptx-sass` 是性能分析的最终依据，`50-performance` 中的瓶颈定位常需下钻到 SASS。

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)

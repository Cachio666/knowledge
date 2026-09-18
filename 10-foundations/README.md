# 10-foundations 数学与数值基础

大模型系统里反复用到、且不理解就会算错的数学与数值前提。

## 本域覆盖什么

| 主题 | 内容 | 状态 |
|---|---|---|
| [floating-point](floating-point/) | IEEE754 格式 / 舍入与误差模型 / 求和与 FMA / 低精度格式 | 已完成 |
| [numerical-linear-algebra](numerical-linear-algebra/) | 范数与条件数 / GEMM 误差 / 分解稳定性 / 迭代法 | 已完成 |
| [performance-models](performance-models/) | roofline 模型 / Amdahl 扩展性 / 实测 roofline | 已完成 |

## 代码约定

本域的程序分两类，区别在于**是否需要真实硬件**：

| 主题 | 语言 | 构建 | 能否在本机跑 |
|---|---|---|---|
| floating-point | Python + numpy | `make run` | ✅ |
| numerical-linear-algebra | Python + numpy | `make run` | ✅ |
| performance-models | Python + numpy（01/02）<br>CUDA（03） | `make py` / `make 11_roofline_probe` | 01/02 ✅<br>03 ❌ 需 nvcc |

前两个主题的内容是 IEEE754 与线性代数的**数学性质**，在 CPU 与 GPU 上完全一致，因此写成可在本机直接运行的 Python，而不是需要远程编译的 CUDA。第三个主题的 `03` 必须实测硬件峰值算力与带宽，只能用 CUDA。

`floating-point/code/fpbits.py` 是 bf16 / fp8 / tf32 的**位运算模拟**模块，不是真实硬件实现——它复现舍入行为，不代表这些格式的吞吐或指令语义。

**本域文档的每个数字都来自对应程序的实际输出。** 核验方法是进入 `code/` 跑 `make run`，把输出与文档里的表格逐行对照。

## 与其他域的关系

- **上游**：无。本域是全库最底层，`prerequisites` 为空。
- **下游**：`20-gpu-architecture`（延迟-带宽模型）、`40-kernels`（roofline 判断算子性质）、`90-training`（混合精度与数值稳定性）、`130-quantization`（误差分析）。
- **横切**：`performance-models` 中的 roofline 模型（正文在本域，回链 `40-kernels`）。

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)

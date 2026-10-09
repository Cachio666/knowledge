# 130-quantization 量化与压缩

用更少的比特表示同样的模型。横切域：既涉及硬件数值格式，也涉及算子实现与推理引擎。

## 本域覆盖什么

| 主题 | 内容 | 状态 |
|---|---|---|
| [numerical-formats](numerical-formats/) | fp16 / bf16 / tf32 / fp8 / int8 / int4 / nvfp4 | 已完成 |
| [theory](theory/) | 量化误差分析 / scale 与 zero-point / outlier 问题 | 已完成 |
| [ptq](ptq/) | 训练后量化 / 校准方法 / GPTQ / AWQ / SmoothQuant | 已完成 |
| [qat](qat/) | 量化感知训练 / 伪量化 / 梯度近似 | 已完成 |
| [quantized-kernels](quantized-kernels/) | 量化算子实现（正文在本域，回链 40-kernels） | 已完成 |
| [kv-cache-quantization](kv-cache-quantization/) | KV cache 量化（★ **正文在本域**，回链 `120-inference/kv-cache`） | 已完成 |
| [sparsity-and-distillation](sparsity-and-distillation/) | 结构化与非结构化稀疏 / 知识蒸馏 | 已完成 |

## 与其他域的关系

- **上游**：`10-foundations/floating-point`（数值格式的误差来源）、`20-gpu-architecture/tensor-core`（低精度算力）。
- **下游**：`120-inference`（量化是推理加速的主要手段）。
- **横切**：量化算子实现（正文在本域，回链 `40-kernels`）；KV cache 量化（★ **正文在本域**，回链 `120-inference/kv-cache`）。

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)

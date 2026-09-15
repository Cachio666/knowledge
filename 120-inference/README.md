# 120-inference 推理引擎与部署

从权重到服务。训练追求吞吐，推理要同时面对吞吐、延迟与显存三重约束。

## 本域覆盖什么

| 主题 | 内容 |
|---|---|
| kv-cache | KV cache 原理与显存核算 / PagedAttention / 前缀共享 |
| batching-and-scheduling | continuous batching / chunked prefill / 调度策略 |
| decoding-strategies | beam search / sampling / speculative decoding / 并行解码 |
| engines | vLLM / TensorRT-LLM / SGLang / llama.cpp 架构对比 |
| serving-infra | 部署形态 / PD 分离 / 扩缩容 / 可观测性 |

## 与其他域的关系

- **上游**：`80-model-architectures`（KV cache 大小由架构决定）、`40-kernels`（算力上限由 kernel 决定）、`60-compilers`。
- **下游**：无。这是栈的最上层。
- **横切**：KV cache 量化（正文在 `130-quantization`，此处回链）。

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)

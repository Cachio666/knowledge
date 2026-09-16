# 40-kernels 算子内核工程

把原理落成可跑的 kernel，并让它跑到硬件允许的上界。

## 本域覆盖什么

| 主题 | 内容 | 状态 |
|---|---|---|
| [gemm](gemm/) | 朴素 → 共享内存分块 → 寄存器分块 → tensor core 与异步流水 → persistent kernel 与 tcgen05 → split-K 与 Stream-K | 已完成 |
| [attention](attention/) | attention 的计算形态 / flash attention 与在线 softmax / flash attention 反向 / warp specialization 与变体 / 解码阶段 kernel | 进行中 |
| [reduce-scan](reduce-scan/) | 树形归约 / warp shuffle 与多级归约 / Hillis-Steele 与 Blelloch 扫描 / 跨 block 与确定性 | 未开始 |
| [softmax-normalization](softmax-normalization/) | 数值稳定 softmax / 在线 softmax 与重标定 / LayerNorm·RMSNorm 与 Welford | 已完成 |
| [elementwise-fusion](elementwise-fusion/) | 访存受限算子的上界 / 融合与流量 / 向量化访存与 grid-stride | 未开始 |
| [moe-kernels](moe-kernels/) | MoE 的计算形态 / grouped GEMM / 路由与 permute·unpermute | 未开始 |

## 代码约定

本域的交付物是 **CUDA 内核**，但写作环境（macOS）**没有 CUDA 工具链**。
因此 `code/` 采用**双轨**形态：编号程序是内核本体，另有一个本机可跑的验证模型。

| 文件 | 性质 | 构建 | 能否在本机跑 |
|---|---|---|---|
| `NN_*.cu` | 编号程序，**与文档一一对应** | `make run` | ❌ 需 nvcc + GPU |
| `verify.py` | 非编号辅助模块 | `make verify` | ✅ 只需 python3 + numpy |
| `specs.py` | 非编号辅助模块，硬件规格常量 | — | ✅ 被 `verify.py` 读 |

```bash
cd <主题>/code
make verify     # 本机可跑：验证算法正确性与代价模型
make run        # 需远程 Linux：编译并运行全部 .cu 内核
```

### `verify.py` 做什么、不做什么

**做**（三类断言，缺一不可）：

| 类 | 内容 |
|---|---|
| **A 算法正确性** | 用 numpy 写参考实现，与文档所述算法逐元素比对，打印最大误差 |
| **B 代价模型** | 按文档公式算 SMEM 占用 / 寄存器估算 / 访存字节 / 算术强度 / 与 roofline 拐点的比值，并给出判定 |
| **C 边界与反例** | 文档 §4 声称「会出问题」的输入，必须**真的跑出问题**——断言的是失败行为成立 |

**不做**：不碰硬件。`verify.py` 跑的是算术与 numpy，**它的产出是推导，不是实测**。

### 共享事实源 `specs.py`

代价模型需要硬件常量（峰值算力、HBM 带宽、SMEM 上限等），这些放在 `specs.py`：

- 每条常量带一行 `# 来源：` 注释与**口径注释**（方向、稠密性、精度与累加器、封装档）；
- **常量名编码口径**，使误用可见（`H100_FP16_TFLOPS_DENSE` 而非 `H100_PEAK`）；
- 只放从来源抄来的数，**不放推导值**；找不到可靠来源的项不写进来；
- 常量与来源**沿用 `20-gpu-architecture/*/code/specs.py` 已有的口径**，
  `code/` 自包含所以重新抄一份，但来源注释必须一致。

### 架构分层

| 档位 | Makefile 目标 | 内容 |
|---|---|---|
| sm_80 可移植 | `TARGETS`（进 `make` / `make run`） | 朴素 / 分块 / 寄存器分块 / `cp.async` / warp shuffle |
| sm_90a | `make hopper` | `wgmma`、warp specialization、TMA |
| sm_100a | `make blackwell` | `tcgen05`、TMEM |

前沿档**不混进默认 `TARGETS`**——否则在 A100 上 `make` 直接失败。
`sm_90a` / `sm_100a` 是 "accelerated" 变体，为它编出的 cubin **跑不了 `sm_90`**。

### 本域数字的口径

| 类别 | 判定 |
|---|---|
| **规格值** | 来自 `specs.py`，逐条带来源 |
| **推导值** | 由公式算出，可在 `make verify` 里复核 |
| **引用值** | 外部来源的实测，须注明「本仓库不复核其测量口径」 |

**本域文档禁止出现「实测」。** 本仓库没有运行过这些内核，`.cu` 的运行结果一律没有。
因此本域文档的 §5「实测数据」**整节省略**，从 §4 直接进「编译与运行」小节。
若某篇确实引用了外部实测数据（如某论文报的加速比），必须标注为**引用值**。

## 与其他域的关系

- **上游**：`20-gpu-architecture`（硬件行为）、`30-gpu-programming`（编程工具）、`10-foundations`（roofline 与数值）。
- **下游**：`60-compilers`（编译器要生成的正是这些 kernel）、`120-inference`（引擎的算力上限由这些 kernel 决定）。
- **横切**：量化算子实现（正文在 `130-quantization/quantized-kernels`，此处回链）；
  MoE 的 all-to-all 通信量（正文在 `110-communication/collectives`，本域只讲 kernel 侧的本地重排）；
  softmax 的数值性质（正文在本域 `softmax-normalization`，`attention` 回链此处）；
  tensor core 的指令形状与 fragment 布局（正文在 `20-gpu-architecture/tensor-core`，本域只回链）。

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)

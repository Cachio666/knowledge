# 20-gpu-architecture GPU 体系结构

硬件的静态结构。不理解这一层，后面所有「为什么这个写法快」都没有依据。

## 本域覆盖什么

| 主题 | 内容 | 状态 |
|---|---|---|
| [execution-model](execution-model/) | SM 内部结构 / warp 与 SIMT / 分支发散 / 占用率与延迟隐藏 | 已完成 |
| [memory-system](memory-system/) | 存储层次 / 访存合并 / 共享内存 bank 冲突 | 已完成 |
| [tensor-core](tensor-core/) | MMA 原语与 tile 形状 / fragment 布局 / wgmma 与流水 / tcgen05 与 TMEM / 2:4 结构化稀疏 | 已完成 |
| [interconnect](interconnect/) | 带宽口径 / 拓扑与 bisection / 拓扑受限的集合通信 / 标称与可达的差距 | 已完成 |
| [architecture-evolution](architecture-evolution/) | 跨代横向对比与规律提炼。**规格明细归各机制主题**（如逐代 SM 规格在 `execution-model`），本主题只做横向比较与趋势归纳，不重复罗列规格 | 已完成 |

## 代码约定

本域的程序分两类，分界依据是**该主题的可验证内容是否需要真实硬件**：

| 主题 | 语言 | 构建 | 能否在本机跑 |
|---|---|---|---|
| execution-model | CUDA | `make run` | ❌ 需 nvcc |
| memory-system | CUDA | `make run` | ❌ 需 nvcc |
| tensor-core | Python + numpy | `make run` | ✅ |
| interconnect | Python + numpy | `make run` | ✅ |
| architecture-evolution | Python + numpy | `make run` | ✅ |

前两个主题讲的是**具体 kernel 在具体硬件上的行为**，只能用 CUDA 实测；后三个主题讲的是**硬件规格的算术与组合**（tile 形状、lane→元素映射、带宽口径换算、bisection 下界、逐代比值），程序从规格常量出发推导，可在任意机器上复核。

### 共享事实源 `specs.py`

后三个主题的 `code/` 各含一个 **`specs.py`**（无编号辅助模块，不计入「文档数 = 编号源文件数」的核对）：

- **每条硬件规格常量都带一行 `# 来源：` 注释**与口径注释（方向、稠密性、精度与累加器、封装档）。
- **常量名编码口径**，使误用可见——用 `NVLINK_PER_GPU_BIDIR_GBps` 而不是 `NVLINK_BW`。
- **只放从来源抄来的数，不放推导值。** 所有推导在编号程序里做。
- **找不到可靠来源的项不写进来**，改记为 `UNSOURCED` 并写「官方未披露」。

改规格只改这一处。

### 本域数字的口径

本域文档的数字只有两类，**禁止写成「实测」**：

| 类别 | 判定 |
|---|---|
| **规格值** | 来自 `specs.py`，逐条带来源 |
| **推导值** | 由规格常量代入公式算出，可在对应程序里复核 |

本域的程序跑的是算术与组合，**不产出实测值**。唯一的例外是引用外部来源的实测数据（如稀疏加速比），必须显式标注「引用值，本仓库不复核其测量口径」。

## 与其他域的关系

- **上游**：`10-foundations`（延迟-带宽模型是理解存储层次的工具；roofline 模型是 `architecture-evolution` 拐点分析的定义出处）。
- **下游**：`30-gpu-programming`、`40-kernels`、`50-performance`、`110-communication`。
- **横切**：NVLink 带宽与拓扑（正文在本域，回链 `110-communication`）；collectives 的 ring 通信量与饱和（正文在 `10-foundations/performance-models`，本域 `interconnect/03` 只讲拓扑受限的部分）。

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)

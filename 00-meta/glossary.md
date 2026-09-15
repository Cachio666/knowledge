# 术语表

全库术语以此为准。新增术语按所属域插入对应分组。

## GPU 硬件与执行模型

| 英文 | 中文 | 说明 |
|---|---|---|
| SM (Streaming Multiprocessor) | 流式多处理器 | GPU 的基本计算单元，内含多个 CUDA core、warp 调度器、寄存器文件与共享内存。是 warp 调度的容器。 |
| warp | 线程束 | 32 个线程组成的调度单位。SM 以 warp 为最小单位调度与发射指令。 |
| SIMT (Single Instruction, Multiple Threads) | 单指令多线程 | 硬件负责将标量写法向量化的执行模型，与 SIMD 的"程序员负责向量化"相对。 |
| thread block | 线程块 | 由多个 warp 组成，是共享内存与同步的作用域，且必须整体调度到同一个 SM 上。 |
| grid | 网格 | 一次 kernel 启动的全部线程块集合。 |
| CUDA core | CUDA 核心 | SM 内执行整数/浮点运算的流水线。其数量常被误当作唯一性能指标。 |
| warp scheduler | 线程束调度器 | SM 内负责挑选就绪 warp 并发射指令的部件。每个 SM 通常有多个。 |
| register file | 寄存器文件 | SM 内寄存器总量。是分配预算，不是缓存，用满即限制占用率。 |
| occupancy | 占用率 | SM 上实际常驻 warp 数与最大可常驻 warp 数之比。受寄存器、共享内存、block 数、warp 数四项约束取最小。 |
| latency hiding | 延迟隐藏 | 通过维持足够多的并发 warp，在部分 warp 等待访存时让其他 warp 继续执行，从而掩盖延迟。 |
| SIMT divergence | 分支发散 | 同一 warp 内线程走不同控制流路径，导致路径串行执行。 |
| reconvergence | 重汇聚 | 发散的分支路径执行完毕后，warp 内线程重新同步到同一控制流点。 |
| predication | 谓词执行 | 用指令前缀谓词代替跳转的机制，可避免短分支产生发散。 |
| ILP (Instruction-Level Parallelism) | 指令级并行 | 单个线程内多条无依赖指令的重叠执行。 |
| TLP (Thread-Level Parallelism) | 线程级并行 | 通过多 warp 并发来填补流水线气泡。 |

## 存储层次

| 英文 | 中文 | 说明 |
|---|---|---|
| memory hierarchy | 存储层次 | 按容量、带宽、延迟权衡分级的存储结构。 |
| HBM (High Bandwidth Memory) | 高带宽显存 | 片外显存，容量最大、带宽最高、延迟最长。 |
| L2 cache | 二级缓存 | 片上全局共享缓存，所有 SM 可见，是 HBM 之前的最后一级。 |
| L1 cache | 一级缓存 | 每 SM 私有，与共享内存共享同一块物理存储。 |
| shared memory | 共享内存 | 片内可编程管理的存储，延迟约比 HBM 低两个数量级。软件管理的缓存。 |
| local memory | 本地内存 | 寄存器溢出时的落点，物理上位于片外，按线程私有寻址。名字有误导性。 |
| coalescing | 访存合并 | 同一 warp 内多个线程的访存请求被合并为更少的内存事务。 |
| cache line | 缓存行 | L2 的传输单位，通常 128 字节。 |
| sector | 扇区 | L1 的传输单位，通常 32 字节。每个 cache line 含 4 个 sector。 |
| bank conflict | bank 冲突 | 同一 warp 内多个线程访问共享内存的同一 bank 的不同地址，导致访问串行化。 |
| padding | 填充 | 通过在共享内存数组维度上加 1 来错开 bank 映射，消除冲突。 |
| swizzle | 重排 | 以异或等方式重映射地址，消除冲突或提升合并度，且不浪费容量。 |
| bandwidth | 带宽 | 单位时间可传输的数据量。衡量访存受限算子的核心指标。 |
| arithmetic intensity | 算术强度 | 每字节访存所对应的浮点运算次数。roofline 模型的横轴。 |
| roofline | 屋顶线 | 以算术强度为横轴、可达性能为纵轴的性能上界模型。 |

## 互联与多卡

| 英文 | 中文 | 说明 |
|---|---|---|
| NVLink | — | NVIDIA 的 GPU 间高速互联，带宽显著高于 PCIe。 |
| NVSwitch | — | 连接多张 GPU 的交换芯片，构成全互联拓扑。 |
| PCIe | — | 通用外设互联总线，GPU 与主机通信的主要通道。 |

## 精度与量化

| 英文 | 中文 | 说明 |
|---|---|---|
| FP32 / FP16 / BF16 | 单/半精度浮点 | 标准浮点格式。BF16 牺牲尾数换取与 FP32 相同的指数范围。 |
| TF32 | — | Tensor Core 上的 19 位浮点输入格式（1+8+10），兼顾精度与吞吐。只在矩阵乘的输入端生效，代码里没有类型签名提示。 |
| FP8 | 8 位浮点 | 分 E4M3 与 E5M2 两种变体，用于 Hopper 及以后的低精度计算。 |
| INT8 / INT4 | 8/4 位定点 | 量化推理常用格式。 |
| PTQ (Post-Training Quantization) | 训练后量化 | 不重新训练，仅用校准数据确定量化参数。 |
| QAT (Quantization-Aware Training) | 量化感知训练 | 训练时插入伪量化节点，让模型适应量化误差。 |
| scale / zero-point | 缩放因子/零点 | 仿射量化的两个参数，决定浮点与整数域的映射。 |
| outlier | 离群值 | 数值分布中的极端值，是激活量化困难的主因。 |

## 数学与数值基础

| 英文 | 中文 | 说明 |
|---|---|---|
| unit roundoff | 单位舍入误差 | `u = 2^-p`（p 为有效精度位数），误差模型 `fl(x) = x(1+δ), \|δ\| ≤ u` 里的那个界。与 `eps = 2^-m` 差 2 倍，混用会让误差估计错一倍。 |
| ulp (unit in the last place) | 末位单位 | 相邻两个可表示数之间的距离。在**同一 binade 内恒定**，因此相对间距 `ulp(x)/x` 在 `(u, eps]` 之间滑动。 |
| binade | — | 区间 `[2^e, 2^(e+1))`。ulp 的绝对值只取决于它落在哪个 binade。 |
| subnormal | 次正规数 | 指数域全 0、隐藏位为 0 的数，填补 0 与最小规格化数之间的空洞。代价是相对精度从 u 一路恶化到 1/2，低于最小次正规数的一半即直接归零。 |
| catastrophic cancellation | 灾难性抵消 | 两个相近量相减，绝对误差不变而相对误差爆炸。减法本身是精确的（Sterbenz 引理），被放大的是输入里已有的误差。 |
| FMA (fused multiply-add) | 融合乘加 | `a×b+c` 只舍入一次而非两次，误差界从 2u 降到 u。 |
| Kahan summation | Kahan 补偿求和 | 用补偿项回收每次舍入丢掉的低位，误差基本不随 n 增长。 |
| condition number | 条件数 | `κ(A) = ‖A‖·‖A⁻¹‖`，输入相对误差传到输出的最大放大倍数。**是紧界**（最坏方向上可达到）但**随机扰动撞不上**（中位放大恒为 1）。依赖范数，报 κ 必须带范数下标。 |
| growth factor | 增长因子 | 分解过程中 `max\|U_ij\| / max\|A_ij\|`，衡量算法引入的膨胀。部分主元 LU 实测中位 2~8。 |
| iterative refinement | 迭代精化 | 低精度分解 + 高精度残差修正，把精度问题转换成「`κ·u` 够不够小」。是「低精度计算 + 高精度主权重」的数值原型。 |
| arithmetic intensity | 算术强度 | `I = FLOPs / Bytes`，判断算子受限于算力还是带宽的唯一自变量。Bytes 只算必须往返 HBM 的字节。 |
| roofline model | roofline 模型 | 以算术强度为横轴、可达算力为纵轴的两段线：`min(P_peak, B_peak·I)`，拐点在 `I* = P_peak/B_peak`。 |
| memory-bound / compute-bound | 带宽受限 / 算力受限 | `I < I*` 时时间被访存主导，提高算术强度能线性提升性能；`I > I*` 时算力封顶。 |
| Amdahl's law | Amdahl 定律 | 串行分数 s 决定加速比上限 `1/s`。固定的是问题规模。 |
| Gustafson's law | Gustafson 定律 | 固定单处理器工作量时，加速比 `a + (1-a)N` 线性无界。与 Amdahl 不矛盾，只是固定的量不同。 |
| split-K | — | 把 GEMM 的 K 维切分给多个 block 再归约。提高小 M/N 时的并行度，代价是累加顺序不确定、结果不可复现。 |

## 训练与推理系统

| 英文 | 中文 | 说明 |
|---|---|---|
| autograd | 自动微分 | 依据计算图自动求导的机制。 |
| KV cache | 键值缓存 | 自回归解码时缓存历史 token 的 K/V，避免重复计算。 |
| continuous batching | 连续批处理 | 在请求级别动态进出批次，而非等待整批完成。 |
| PagedAttention | 分页注意力 | 将 KV cache 按页管理，消除显存碎片。 |
| speculative decoding | 投机解码 | 用小模型草拟、大模型验证，减少解码步数。 |
| ZeRO | — | 将优化器状态、梯度、参数分片到各卡的显存优化方案。 |
| FSDP (Fully Sharded Data Parallel) | 全分片数据并行 | ZeRO 思想在 PyTorch 中的实现。 |
| all-reduce | 全规约 | 集合通信原语：所有参与者的数据求和后广播给所有人。 |

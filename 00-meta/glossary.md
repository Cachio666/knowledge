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
| MMA (Matrix Multiply-Accumulate) | 矩阵乘加 | Tensor Core 执行的基本指令族。形状固定（如 `m16n8k16`），**「固定形状」是理解它的钥匙**——tile 尺寸、fragment 布局、操作数来源都由它决定。 |
| fragment | 片段 | 一个矩阵被**分布式**存在一个 warp 的 32 个 lane 寄存器里的形态。只有 `mma.sync` 的 lane→元素映射有规范定义（可推导、可穷举验证双射）；**`nvcuda::wmma` 层的映射官方明确写 "unspecified and target architecture dependent"，不可推导**。 |
| ldmatrix | — | 把共享内存里的一块矩阵一次搬进 fragment 的指令。fragment 布局中 `groupID = lane>>2` / `threadID_in_group = lane%4` 的物理来源就是它（一个 8×8 的 16-bit 矩阵每行 16 字节由 4 个连续 lane 各取 4 字节）。 |
| warpgroup | 线程束组 | 4 个连续 warp（128 线程）。Hopper 起 `wgmma` 的执行单元从 warp 升到 warpgroup，且为异步。 |
| wgmma | — | Hopper (sm_90a) 的异步 warpgroup MMA。**B 操作数必须在共享内存**，A 可以是寄存器或共享内存描述符——别写成「两者都必须在 SMEM」。 |
| TMEM (Tensor Memory) | 张量内存 | Blackwell 引入的片上存储，**512 列 × 128 行 × 32-bit = 256 KB**（官方口径是 **per CTA**，不是 per SM）。它让累加器不再占寄存器文件（两者恰好同为 256 KB）。需要一个 warpgroup 才能覆盖全部 128 lane。 |
| tcgen05 | — | Blackwell (sm_100a) 的 MMA 指令族。与前代的关键差异是**单线程发射**（不再是 warp 或 warpgroup），且 **D 累加器必须在 TMEM**。 |
| Tensor Core | 张量核心 | 做 MMA 的专用单元。**「每 SM 几个 tensor core」不是可靠指标**——NVIDIA 每代都在改「一个 tensor core」的定义，数量从 Volta 的 8 降到 Ampere 的 4 而算力翻倍。稳定的量是 **FP16 FMA/clock/SM**。 |
| warp shuffle | 线程束内交换 | `__shfl_down_sync` / `__shfl_up_sync` / `__shfl_xor_sync`，在 warp **内部**按 lane 索引搬运寄存器值。**永不跨 warp**（与 `width` 无关）。`width` 改变的是 warp **内部**的段划分；源 lane 越出本段时**静默返回调用者自己的值**——后果是归约结果**无声偏大**，且触发条件是 `offset ≥ width`，倍数**恰为 `2^超界步数`**。写成「跨 warp 读」会让排错方向整个跑偏。 |
| `__syncwarp` | 线程束内屏障 | warp 内交换必须用它，**不能用 `__syncthreads()`**。后者是**块级**屏障，其到达计数在 PTX 层面按 warp 计（一条 lane 到达即整个 warp 记到），因此**不能强制已发散的 lane 收敛**，也**不提供 warp 内 lane 之间的顺序保证**。分歧在**两代硬件上是相反的失败模式**：Volta 之前按 warp 记账 → 可能**提前释放**（静默读到未写完的共享内存）；Volta 起按线程强制 → **挂死**。 |

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
| kernel fusion | 算子融合 | 把一串逐元素算子并进一个 kernel，让中间结果**不落 HBM**。字节从 `2kN·esize` 降到 `2N·esize`，**上界是链长 `k`**——但这个上界**只在「每个中间结果都往返 HBM」时成立**：中间数组命中 L2 时，不融合的流量本来就≈ `2N`，收益趋于 **1×**。所以真实比值落在 `(1, k)` 开区间，位置由 `(k+1)·N·esize` 与 L2 容量决定。**常见误写是「上界趋于 2」或「融合越多越快」。** |

## 互联与多卡

| 英文 | 中文 | 说明 |
|---|---|---|
| NVLink | — | NVIDIA 的 GPU 间高速互联，带宽显著高于 PCIe。**厂商标称的每 GPU 带宽全部是「双向合计」**：H100 的 900 GB/s = 450 发送 + 450 接收，**任何单向搬运的天花板是 450**。做「搬运 X 字节要多久」的推导时用错会差 2 倍。 |
| NVSwitch | — | 连接多张 GPU 的交换芯片，构成**全互联（full bisection）**拓扑。它的价值不只是「更宽」，而是**路径均匀性**——任意 GPU 对都能拿到满带宽，而 ring 下只有相邻对能。 |
| PCIe | — | 通用外设互联总线，GPU 与主机通信的主要通道。**一个「带宽」有三种口径**：线速率（`GT/s × lane ÷ 8`，不可达）、载荷（扣编码）、营销取整。8b/10b 只用于 Gen1/2，**Gen3 起改用 128b/130b**，所以「标称 vs 有效」的差额从前两代的 20% 掉到 1.5%。**NVIDIA 自己用的是标称值。** |
| bisection bandwidth | 对分带宽 | 把网络均切成两半时，**跨越切口的链路总容量**。ring 的 bisection **不随 N 增长**（恒为 `L × b`），全互联随 `N²` 增长。集合通信的下界由它给出。官方直述值：HGX A100 8-GPU = 2.4 TB/s，HGX H100 8-GPU = 3.6 TB/s（恰为聚合端口带宽的一半）。 |
| NVLink Domain | NVLink 域 | 官方定义是「能通过 NVLink 互相通信的节点集合」，**是跨 OS 实例的概念**。逐代：DGX-1 8 → DGX-2 16 → HGX A100/H100 8 → H100 + NVLink Switch System 256 → GB200 NVL72 72 → 多机架 576。**「A100 SuperPOD 的 256 卡」不是 NVLink 域**（跨节点走 InfiniBand，域仍是 8）。 |
| P2P (peer-to-peer) | 对等访问 | GPU 之间直接走 PCIe 交换数据，不经主机内存。**失败是静默的**——PCIe peer 写是无 ACK 的 posted transaction，丢包只体现在 host kernel log，GPU 侧看到的是 hang 或静默数据损坏。**ACS 与 IOMMU 都会破坏它**（Grace Hopper/DGX 平台例外，应保持 IOMMU 开启）。 |
| algbw / busbw | 算法带宽 / 总线带宽 | nccl-tests 的两个指标。`busbw = algbw × 修正系数`（AllReduce 是 `2(n-1)/n`，ReduceScatter/AllGather/AlltoAll 是 `(n-1)/n`，Broadcast/Reduce 是 1）。**只有 busbw 能与硬件峰值对比，且上限是单向的那一列（H100 = 450 GB/s），不是 900。** 但它在 NVLS/SHARP 下会高估。 |
| all-to-all | 全交换 | 每个 rank 把数据分给所有其他 rank，没有归约可做。跨切口量 `∝ N`（all-reduce 是常数 `2S`），**跨界占比恒为 1/4 而不随 N 下降**——所以它是 bisection-bound 的，规模越大越吃拓扑。 |
| scale-up / scale-out | 纵向 / 横向扩展 | 两套**并行**网络，不可合并。GB200 NVL72 的 130 TB/s 是 scale-up（NVLink 机架内），scale-out 是 72 × 800 Gb/s = 7.2 TB/s，**两者差 18 倍**。 |
| InfiniBand | — | 机间互联。EDR 100 / HDR 200 / NDR 400 / XDR 800 Gb/s，按 `Gb/s ÷ 8` 换算是**未扣协议开销的上界**。节点内 vs 节点间每 GPU 双向带宽之比：H100 是 18:1，**按单向比则是 9:1——两个数都对，取决于表达什么**。 |

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
| 2:4 structured sparsity | 2:4 结构化稀疏 | **每连续 4 个元素（沿 K 维）中恰有 2 个非零**，逐行约束。只施加在 A 操作数上（B 与累加器保持稠密），所以是 **SpMM 不是 SpGEMM**。理论加速 2×，端到端实测普遍只有 **1.2×~1.6×**。 |
| metadata (稀疏) | 稀疏元数据 | 记录每个 4 元素组中两个非零位置的索引，**每个 4 元素组用两个 2-bit 索引 = 共 4 bit**（不是 2 bit）。16-bit 数据下对原数据占 1/16，对压缩后载荷占 1/8。**含 metadata 的总压缩率是 36/64（省 44%），不是 50%**。 |
| dense / sparse (算力口径) | 稠密 / 含稀疏 | 厂商标称的 TFLOPS **默认已含 2:4 稀疏的 2×**（如 H100 的 1979 是 sparse，dense 只有 989.5）。**更麻烦的是书写顺序：NVIDIA 自家文档 dense\|sparse 与 sparse\|dense 两种都用了**（Blackwell 技术简报用前者，DGX B200/B300 datasheet 用后者）。永远看脚注，别信顺序。 |

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
| work / depth | 工作量 / 深度 | 并行算法的两个**独立**成本度量：work 是总操作数，depth 是关键路径长度。Brent 模型 `T = work/p + depth` 里 **p 越小 work 项越主导**，所以「谁的 work 小」在低并行度下赢、「谁的 depth 小」在高并行度下赢。 |
| Hillis-Steele scan | — | Kogge-Stone 形态的并行前缀和：work `n·log n`、**depth `log n`**。work 大但步数少。 |
| Blelloch scan | — | Brent-Kung 形态（up-sweep + down-sweep）：work `2n−2`、**depth `2·log n`**。**两者 depth 都是 Θ(log n) 但差一倍，不是「相同」**——临界并行度 `p* = n·(1 − 2/log n)` 逼近元素数 `n` 本身，即**只有在并行度几乎等于元素数时 Hillis-Steele 才反超，且优势只有 5%~10%**。 |
| online softmax | 在线 softmax | 用**运行最大值 `m` 与运行和 `l`** 递推，把 softmax 从「需要先看完整行」变成**一遍流式**：`m_new = max(m_old, 块内 max)`、`l_new = l_old·exp(m_old−m_new) + Σexp(x−m_new)`、`O_new = O_old·exp(m_old−m_new) + P·V`。合并算子 `⊕` 构成**半群**，任意分组/顺序给同一数学结果——这是分块流水与 attention 能分块的依据。⚠ **`m_old ≤ m_new` 恒成立**，rescale 因子恒 ≤ 1，写反会溢出。 |
| Welford 算法 | — | 单遍算均值与方差：`mean += d/n`、`M2 += d·(x−mean_new)`。**它真正的比较对象是 `Σx²−nμ²`**（后者在大均值下灾难性抵消、会算出**负方差**），**不是两遍法**——大均值下两遍法反而略优（Welford 的均值是逐项递推，会撞上「更新量被 `|mean|` 的 ulp 吞掉」）。 |

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
| flash attention | — | 用在线 softmax 分块，**不物化 `S×S` 矩阵**。收益不是「提高了算术强度」，而是**绝对字节数从 O(S²) 降到 O(S·d)**（S=16384 时 65 倍）。⚠ 朴素 attention 的 AI **渐近于 `d` 且随 S 上升**，不要照抄「S² 太大所以 memory-bound」。 |
| GQA / MQA | 分组/多查询注意力 | 多个 query 头共用一份 KV，**群组大小 `r = heads_q/heads_kv`**（文献惯例；MHA 时 `r=1`，MQA 是 `heads_kv=1` 即 `r=heads_q`）。decode 阶段**算术强度在数值上就等于 `r`**（fp16），**与序列长度和 head 维度都无关**——所以长上下文只让绝对时间线性变长，不改变瓶颈性质。 |
| MoE (Mixture of Experts) | 混合专家 | 每 token 只过一个子集的专家（`top_k` 个）。计算量 `4·tokens·top_k·d_model·d_ff`（**两个 GEMM**：up + down；SwiGLU 三投影则是 6）——**与专家总数无关**。⚠ **省的是 FLOPs，不是显存**：全部专家的参数都要驻留，**权重流量随专家数线性上涨**，所以权重流量的算术强度 `2·T·top_k/(E·esize)` 随 `E` **反比下降**，拐点批大小 `T* = I*·E·esize/(2·top_k)` 随 `E` **线性上涨**。 |
| grouped GEMM | 分组矩阵乘 | 把变长的一批小 GEMM 合成一次调用，**N、K 相同、只有 M 变长**——所以问题归结为「M 不规整」。**按实际长度分组**的浪费只来自「`M_e` 不是 `BM` 的整数倍」（与分布无关）；**按最大专家填充**的浪费是 `1 − 均值/最大值`（均匀 ~4%、幂律 >60%）。⚠ 网格调度比填充策略重要得多：**一个专家一个 block** 只填满 `E/槽位数` 的机器，**一个 tile 一个 block** 能到 99%。 |
| 容量因子 (capacity factor) | — | 每个专家开出的 slot 数上限。**定义与语义的正文在 `80-model-architectures/moe`**，kernel 侧只关心它带来的三种状态：正常、被截断（要 mask）、以及**`k` 个选择全被截断**——最后这种 token 前向输出恰为 0、反向梯度也恰为 0，**在本步训练里完全不存在**。 |
| IR (Intermediate Representation) | 中间表示 | 编译器内部表示程序的形式，决定了 pass 能看见什么、能改什么。本域出现的两种具体形态：MLIR 的四层递归结构（Operation / Region / Block / Value）与 XLA 的 HLO。 |
| SSA (Static Single Assignment) | 静态单赋值 | **每个值恰好被定义一次**，于是「值」与「定义它的那条指令」一一对应——这是「形状可以挂成指令自身的属性、不必另建值表」的前提。同一个值被定义两次不是风格问题：def-use 分析整个失效。 |
| dialect (方言) | 方言 | MLIR 用命名空间区分**抽象层次**的机制：`arith.addf`、`linalg.matmul`、`scf.for`、`gpu.launch` 分属四个方言。**区分层次的不是结构，是 operation 的名字**——结构（Operation / Region / Block / Value）全库共用一套。 |
| lowering / progressive lowering | 降级 / 渐进式 lowering | 把 IR 从高层表示翻译到低层表示。**关键在「渐进」**：不存在一次从顶到底的转换，存在的是一串各自只做一件小事的变换（可单独测试、单独替换），代价是**每一步都有前置条件**，且不满足时的失败方式反直觉——`linalg` 降成 `scf.for` 要求操作数已是 memref 语义，只要还有 tensor 就 "definitely fails"。 |
| bufferization | 缓冲化 | 把 tensor 语义的 op 转成 memref 语义的 op。**它不是一次改名**：它同时决定了哪些中间结果真的要被分配出来——「物化次数」因此变成一个可数的量（数 `memref.alloc` 的个数）。 |
| tensor semantics / memref semantics | tensor 语义 / memref 语义 | `tensor<2x3xf32>` 与 `memref<2x3xf32>` 的形状部分完全相同，差别只有一处：**存储有没有被指定**。memref 是「指向内存的**引用**」（官方定义 *Shaped reference to a region of memory*），因此**两个 memref 可以指向同一片内存**，tensor 值不能——别名是全部危险的来源。⚠「值语义 / 缓冲语义」讲得通但**不是官方引文**，引用时只引上面那两个词。 |
| compute / schedule separation | 计算与调度分离 | 「算什么」（`@T.prim_func`）与「中间结果放不放到独立存储里」（`tvm.s_tir.Schedule`）分成两层。**默认行为反直觉：每个中间结果都被物化在根作用域**，不是被 inline 掉。 |
| materialize / recompute | 物化 / 重算 | 本域贯穿全局的那条主线。**物化 = 为中间结果分配独立的存储**（寄存器、共享内存、全局内存都算）；**重算 = 根本不分配存储**，把计算直接替换进消费者的表达式。每一次 lowering、每一次变换、每一个 pass 都在做这个权衡。 |
| compute_at / compute_inline | — | `s_tir.Schedule` 里管「放到哪儿」的两个方向：`compute_at`（物化，缩到某个循环内）、`compute_inline`（不物化）。**`compute_root` 不在 v0.26.0 的 API 里**——没被搬走的 block 本来就在根位置；TE 时代「三态齐备」是旧口径，不要照抄。 |
| graph break | 图断点 / 切图 | 捕获中途做不到时**把图切断**。切出来的只是**那一段**区域改为跑普通 Python，**前后两段仍然各自被编译成图**——「整段退回 eager」是本主题最贵的误解。断点处仍存活的值成为下一张子图的输入（官方为此生成 resume function），所以切图不是丢掉前半段。官方把它定位成特性（"graph breaks are a feature"）：带断点的程序能跑，拒绝编译的程序不可用。 |
| guard | 守卫 | 挂在 code object 上的一组布尔假设，**编译产物只有在它成立时才能复用**。内容是官方措辞的 **constant values, types, and object IDs**——既有形状与类型，也有**常量的精确值**。 |
| TorchDynamo | — | `torch.compile` 的追踪器，用 CPython 的 **Frame Evaluation API（PEP 523）** 在**每个函数帧进入时**决定该帧编不编。**不是 `sys.settrace`**（那是行级事件，粒度与语义都不同），**也不是 `sys.monitoring`**（与 `torch.compile` 已知不兼容，注册回调会引发 graph break）。 |
| FX graph | FX 图 | Dynamo 逐条模拟字节码、把对象包成 `VariableTracker` 之后产出的图。它是**调度阶段的输入**——图捕获本身不做任何物化决策；它到 Inductor 自己的 IR（`ComputedBuffer` 之类）之间还有一次 lowering。 |
| fusion group | 融合组 | Inductor 调度阶段的产物，由 `SchedulerNode` 划成 `FusedSchedulerNode`。**划法就是「哪些中间结果不落内存」这个决策本身**；判据是**一串门槛**（垂直的索引逐字匹配、水平的字节阈值、全局的节点数与峰值显存门），不是一条规则。 |
| HLO (High Level Operations) | — | XLA 与前端之间的中间表示。两条性质决定了它的形状推断长什么样：**它是 SSA 的**（每条指令恰好产生一个值，形状可挂成指令属性）、且**大部分是纯的**。 |
| layout assignment | 布局指派 | XLA 在**融合之前**跑的阶段：约束对象沿 HLO 图创建、排序、传播，冲突时高优先级者胜，并由 **`kCopy`** 桥接不兼容的布局。⚠ 不要与方向相反的 **memory space assignment** 混为一谈——后者的目标是把值塞进 alternate memory，**且 GPU 上跑的并不是它**。 |
| cost model | 代价模型 | 在少量实测数据上训一个模型，给没测过的配置打分。**衡量它的关键指标不是预测得多准，是排序排得多对**——搜索从来不问「这个程序跑多少毫秒」，只问「这两个里哪个更值得测」。⚠ 拿 RMSE 当合格线是这一行的典型误用：绝对值崩了（R² 变负）而排序仍可用，是常态。 |
| search space | 搜索空间 | 「搜什么」。两级结构：**sketch**（骨架，由推导规则递归生成）+ **annotation**（低层参数，随机采样）。⚠ **融合决策在 sketch 阶段就已定死**，把它读成 annotation 层的一个开关是错的。 |
| nominal / effective size | 名义基数 / 有效基数 | 名义基数 = 把各参数取值范围乘起来得到的数，**是个假象**（大量配置生成同一段代码，另有一批根本编译不过）；有效基数 = **等价类中可行的那部分**。⚠ 有效基数是**等价类的个数，不是配置的个数**；按名义基数估预算，最坏情况下 82% 花在无效点上。 |
| autotuning | 自动调优 | 让机器在搜索空间里挑实现。三个部件相乘：**空间 → 代价模型 → 搜索**。⚠「多测几个候选不会让结果变差」只在候选集不变时成立——**预算单调，空间不单调**，改动候选集本身会让结果变差。 |

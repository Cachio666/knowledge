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
| lane（lane ID） | 通道编号 | warp 内线程的编号，0–31。`mma.sync` 的 lane→元素映射、`shfl.sync` 的 lane 寻址、bank 冲突里「哪个 lane 访问哪个 bank」都建立在它上面。⚠ 本表别处（`fragment`、`ldmatrix` 两条）已经在用这个词而从未定义过。 |
| CTA (Cooperative Thread Array) | 线程块阵列 | **与 `thread block` 同指不同名**（PTX / CUTLASS 文档用这个名字）。⚠ 它不是另一种东西，也不是「块的数组」——`CTA shape` 就是 thread block 的形状。本库正文里两个名字混用。 |
| legacy default stream | 传统默认流 | CUDA 的**隐式**默认流（stream 0）。它与**同一 `CUcontext` 内所有其它流**同步，用 `cudaStreamNonBlocking` 建的流除外。⚠ 「同步」的两半不等价：入边数 **O(流数)**、出边数 **O(后续动作数)**——它是一条**星形栅栏**，不是「两条阻塞流之间互相同步」。 |
| stream capture | 流捕获 | 把一段流上动作**录成**一张图（`cudaGraph_t`）的过程，之后可实例化并重放。⚠ capture 期间被捕获的流**不是一条能执行的队列**——「不许做的事」（同步、落在 legacy 流上的动作等）一违规即 `cudaErrorStreamCaptureInvalidated`，且**一次违规废掉的是所有关联的图**。 |
| CUDA Graph | — | 把宿主侧的**发起动作**录成一张图、一次提交的机制。⚠ 它省的是**宿主侧动作数**，**不是** kernel 执行次数——`8:1` 这类比值是动作数之比，**不能读成「加速倍数」**。 |
| PTX (Parallel Thread Execution) | — | NVIDIA 的**虚拟 ISA**，由 `ptxas` 编译成 SASS。PTX ISA 文档是**规范**（可引章节号），`ptxas` 与 SASS 是**某实现**。⚠ PTX 的寄存器是**声明的名字**（`.reg .b32 %r<8>`），不是硬件配额——声明量与实际触及量可以任意不等，且与 SASS 上的物理寄存器数没有可比关系。 |
| SASS | — | GPU 的**物理 ISA**，即 `ptxas` 输出的机器码。⚠ 它是**某实现的产物，不是规范**；本库关于 SASS 的事实一律是转述或来自反汇编，本机无 `nvdisasm`，**一个 SASS 字节都没有观测过**。 |
| block-level program | 块级程序 | Triton 的编程模型：程序作用在**块（张量）**上，而不是单个线程上。⚠ 块内每个元素归哪个线程管**由编译器决定**，源码里不写；`layout conversion` 与「归约跨 warp」这些代价，正是从那个不写的决定里长出来的。 |

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
| 2:4 structured sparsity | 2:4 结构化稀疏 | **每连续 4 个元素（沿 K 维）中恰有 2 个非零**，逐行约束。只施加在 A 操作数上（B 与累加器保持稠密），所以是 **SpMM 不是 SpGEMM**。理论加速 2×；实测要**分两档说**——**层级/内核级约 1.3×–1.7×**，**端到端约 1.2×**，且**小 batch 可能低于 1×**（PyTorch 官方教程 BERT-base/SQuAD bs=4 实测 0.71×，更慢）。**两者是不同口径，不得混称。** |
| metadata (稀疏) | 稀疏元数据 | 记录每个 4 元素组中两个非零位置的索引，**每个 4 元素组用两个 2-bit 索引 = 共 4 bit**（不是 2 bit）。16-bit 数据下对原数据占 1/16，对压缩后载荷占 1/8。**含 metadata 的总压缩率是 36/64（省 44%），不是 50%**。 |
| dense / sparse (算力口径) | 稠密 / 含稀疏 | 厂商标称的 TFLOPS **默认已含 2:4 稀疏的 2×**（如 H100 的 1979 是 sparse，dense 只有 989.5）。**更麻烦的是书写顺序：NVIDIA 自家文档 dense\|sparse 与 sparse\|dense 两种都用了**（Blackwell 技术简报用前者，DGX B200/B300 datasheet 用后者）。永远看脚注，别信顺序。 |
| e8m0 | 8 位纯指数（无符号、0 位尾数） | **没有零、没有负号、没有 Inf**；NaN 是 `0xFF`。bias 127，可表示 2^-127…2^127。**取整是 round-half-away-from-zero**（ONNX 的 `round_mode=up`），不是 RNE。实测 `-1.0 → 1.0`（符号被丢）、`0.0 → 2^-127`（**零无法表示**）。 |
| MX / Microscaling | 微缩放格式族 | OCP 规范（v1.0, 2023-09-07）：**块大小一律 32 个元素共享一个 E8M0 scale**；元素格式 MXFP8/MXFP6/MXFP4/MXINT8。**块 scale 的取整用 Floor（出自 MX 论文 Algorithm 1，不是规范规定）**，代价是 `448/256=1.75<2` 导致块内最大值超上限必须饱和——**溢出块占比：e4m3 阈值下 25%（解析）/16.9%（MC）；e2m1 阈值下 50%/29.6%**（两套阈值不可混）。**元素取整才是 RTNE。** |
| NVFP4 | NVIDIA 的 4 位浮点 | 元素 **E2M1**（幅值上限 ±6）、**块大小 16**、**两层 scale**：`s_block`（FP8 E4M3，每 16 元素）+ `s_global`（FP32，整张量）。与 MXFP4 的三点区别：块 **16 vs 32**；block scale **E4M3（带尾数）vs E8M0（纯 2 的幂）**；**多一层 per-tensor global scale**。 |
| 块缩放 / block scaling | 按块共享 scale | 把张量切成固定大小的块、每块一个 scale。**与 per-channel 的区别是块沿哪一维切。** 代价是 **metadata 的比特开销**（MX 的 e8m0 是 8 bit/32 元素 = 0.25 bit/元素）。⚠ **「4-bit」在不同记账口径下是 4.00 / 4.25 / 4.50 三个不同的整块宽度**（纯载荷 / `32/G` / `64/G`），引用必须说明口径。 |
| eps（浮点格式） | 1.0 处的可表示间距 | 定义为 **`2^-M`**（M 是该格式的尾数位数），**不是抄 `finfo`**。⚠ **`torch.finfo(torch.float8_e5m2fnuz).eps` 报 0.125 是错的，真值是 0.25**——原因是它的 `epsilon()` 照抄了 `Float8_e5m2.h` 的码字 `0x34`，而两个格式 bias 差 1。**这是「`finfo` 是实现、不是规格」的实例。** |
| SQNR | 量化信噪比 | `6.02n + 1.76` 的前提是**满量程正弦**。一般式有两种**各自自洽**的口径：跨度口径 `6.02B + 10.8 − 20log₁₀(Xm/σx)`（`σx = Xm/(2√2)` 时退化）或峰值口径 `6.02B + 4.77 − …`（`σx = Xm/√2` 时退化）。**判据：两种都必须退化到 `6.02B+1.76`，拼错常数会差 6 dB。** ⚠ **浮点格式的 SQNR 与分布几乎无关**（约 1 dB），**定点格式强相关**（可差 20+ dB）——讨论「该法则适不适用」前必须先说清是哪种。 |
| per-tensor / per-channel / per-token / per-group | 量化粒度 | **轴约定必须写死**：按 `[tokens, features]` 排布时，**「每行一个 scale」= per-token**、**「每列一个 scale」= per-channel**（写反不会报错，只会让结论与 KIVI 相反）。**per-token 只治「行异质」、per-channel 只治「列异质」，二者不可互替**；outlier 在列上时 per-channel 比 per-token 多约 20 dB（**该差值依赖张量宽度与 outlier 位置**）。⚠ **per-group 的切分维度在各论文不统一**（Q-BERT 沿输出维、GPTQ/nuQmm 沿输入维、Dettmers 拉平）。 |
| 最优 clipping / ACIQ | 裁剪到比 max 更小的范围 | **裁剪到张量 max 是次优的**：`Δ = 2α/2^M`，α 越大 Δ 越大、舍入噪声 ∝ `α²/(3·2^{2M})`。Laplace 下 ACIQ 解析解 `α* = 2.83b / 3.89b / 5.03b`（M=2/3/4，**论文印值**）。**位宽越低，最优解越倾向牺牲更多饱和换更小的 Δ**（饱和概率 2-bit 5.90% → 4-bit 0.65%）。 |
| 校准 / calibration | PTQ 定 scale 的步骤 | 方法族：min-max、滑动平均（**首次观测不走 EMA**，`c=0.01`）、percentile（**可靠一手答案只有 99.99**）、MSE、entropy/KL（TensorRT）。**★ PyTorch 的 `HistogramObserver` 是 L2 最小化，不是 KL**——docstring 逐字写着 L2 error minimization、跟随 caffe2 的 `NormMinimization`，**行为方向与 TensorRT 的 KL 校准器（激进裁剪离群值）相反**。**把两者混为一谈是本域最常见的错误。** |
| GPTQ | 逐列量化的 PTQ | 用二阶信息（Hessian 逆）做**逐列量化 + 误差补偿**。论文超参：**block size 128 列**、dampening **λ = 对角均值 1%**、校准集 128×2048 token。⚠ **`blocksize`（计算批处理，不改数学结果）≠ `group_size`（量化粒度）**；**`act-order` 不在论文里**（是实现的 `argsort(diag(H))`，按**激活二阶矩**降序），收益高度模型相关。⚠ **`dampening` 默认值库间不一致**（论文 1% vs transformers 0.1）。 |
| AWQ | 激活感知的权重量化 | 核心洞察：只有 **0.1%–1%** 的显著权重，且**识别要看激活分布而不是权重分布**。误差比 = `(Δ'/Δ)·(1/s)`。搜索 `s = s_X^α`，α ∈ [0,1]——**步长只在官方实现里：`n_grid=20`（步长 0.05，α=1.0 取不到）**。 |
| SmoothQuant | W8A8 的激活难度迁移 | 用**等价变换**把激活的量化难度迁到权重：**`s_j = max(|X_j|)^α / max(|W_j|)^{1−α}`**。α 典型 0.5（OPT/BLOOM），甜蜜区 0.4–0.6。**训练无关**（数学等价 + 可离线融合进前一层）。⚠ **它是 8-bit 方法，W4A4 会崩**。⚠ 摘要与正文的收益数字不同（1.56×/2× vs 1.51×/1.96×），引用须标位置。 |
| STE | 直通估计器 | 把硬阈值/取整在反传中当恒等。**有三种变体**：纯恒等（Bengio 2013）、**裁剪 STE**（BNN：越界梯度置零）、XNOR-Net 的恒等+缩放。**★ PyTorch 的伪量化反传是「带掩码的裁剪 STE」**：`grad = grad_out · I[qmin ≤ round_half_even(x/s + zp) ≤ qmax]`——**掩码作用在舍入后的量化码上**，不是 x 的实数区间上。⚠ 原文还有两条与流行转述相反的：**「乘 sigmoid 导数」效果更差**、**只在单层内符号正确**。 |
| LSQ / PACT | 学习量化参数的 QAT | **LSQ 学每层的 step size `s`**（`∂v̂/∂s` 三段分段线性，梯度乘 `1/√(N·Q_P)` 缩放——**不加会不收敛**）；**PACT 学 clip 上界 α**（`∂y_q/∂α` **在 clip 点以下处处为零**）。**三层区别**：学什么 / 梯度形状 / 对 round 的处理（LSQ 保留 round 供下游求导）。 |
| W8A8 / W4A16 | 量化位宽记法 | `W<权重位宽>A<激活位宽>`。W8A8 用 **int32 累加器**（int8×int8 单乘积 ≤16129，**int16 在 K≥3 就溢出**、int32 要到 **K≥133145**），scale 在 **epilogue 融合**应用。W4A16 **不是整数累加**，瓶颈在**写侧带宽**（独立反量化 pass 的写是读的 4 倍）。 |
| 反量化融合 / dequant fusion | 把解包融进 GEMM | 独立的反量化 pass **读 `K·N/2` 字节（int4）却写 `K·N·2` 字节（fp16）**，被**写侧带宽**卡死；融进 GEMM 后 fp16 权重不落回 HBM。⚠ **「Σx 列和修正」在官方文档里只有 W8A8 版本**，主流 W4A16 kernel（Marlin/Machete/ExLlamaV2）**都逐元素做 `(q−z)`**。 |
| KIVI | KV cache 的非对称 2-bit 量化 | **K 用 per-channel、V 用 per-token**——因为 **K 在少数固定 channel 上有大幅值 outlier**（且**持久**：该通道在所有 token 上都有），而 V 没有这种 channel-wise 模式。超参 **G=32、R=128**（**128 是 residual length 不是 group size**）。⚠ **KIVI 全文没有 perplexity 数字**（只报任务准确率）；**「2.35×–3.47×」是吞吐量不是显存压缩**（显存口径是 2.6× less peak memory **including model weight**）；**「15×」是输出误差 Δ 的比，不是重建误差的比**。 |
| QJL | 1-bit 的 KV 量化 | **只把 K 量化到 1 bit**，V 仍走标准 per-token；口径是 **3 bits/FPN**。**「zero overhead」指不需要为每块存 scale/zero（省 1–2 bit/数），不是零计算开销。** 需要 JL 投影、且 query 侧要做同样的投影（asymmetric 的成本）。定理：每位 key 只需 `m ≈ ε⁻²log n` bit，**与嵌入维度无关、只随序列长度对数增长**。 |
| Wanda / SparseGPT | LLM 的 one-shot 剪枝 | 判据：SparseGPT `|W|²/diag((XXᵀ+λI)⁻¹)`、Wanda `|W|·‖X_j‖₂`。**归约方向：`crit_SparseGPT = crit_Wanda²`**（对角 H 下；**取 sqrt 后才恒等**）。**SparseGPT 有权重更新、Wanda 没有**（can be used as is）；**SparseGPT 同时做量化**（式 7），Wanda 不做。速度差 **242×–376×**。 |
| 蒸馏温度与 T² | 软标签蒸馏的标度 | `q_i = exp(z_i/T)/Σexp(z_j/T)`；梯度 `∂C/∂z_i = (1/T)(q_i − p_i)`。**T² 的来源**：软标签产生的梯度幅度按 `1/T²` 缩放，同时用硬/软目标时要乘回 T² 才保持相对贡献。⚠ **「乘 T² 后恒定」不精确**（有漂移，残差 `O(1/T)`）；**收敛极限是 `(z−v−mean)/N`**（零均值是前提），**速率是 `O(1/T)` 不是 `1/T²`**——**「1/T²」说的是幅度缩放，是两个量。** ⚠ **原论文里没有 α 这个符号**（是社区约定，两种写法的换算是 `β = (1−α)/α`，**不是互为倒数**）。 |

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
| speedup | 加速比 | 优化前后的比值。⚠ **必须分列「单 kernel」与「端到端」两种口径**（`45-kernel-lifecycle/README.md` §数字口径）——同一份测量里两个数可以差一个数量级，混用即「同名不同指」。⚠ 本库多处拿到的只是**动作数 / 调用次数**之比，那不是加速比。 |
| ridge point | roofline 拐点 | 算术强度轴上算力线与带宽线的交点，`I* = P_peak / B_peak`。**它不是芯片的一个数，而是「算子 × 精度路径」的联合属性**：H100 上 FP32 向量路径 `20.0000`、FP16 张量核路径 `295.3731`，相差 **14.77 倍**——落在两者之间的算子换个精度路径就得到**相反**的判定。⚠ 峰值必须用 **dense**（H100 FP16 是 989.5，不是产品页的 1979）；可达带宽更低 ⇒ 可达 ridge **更高** ⇒ 判据**更严**（与直觉方向相反）。⚠ 两个来源报不同的 ridge 时，**先比带宽口径**——多数分歧在**分母**上，不在算力上。 |
| latency-bound | 延迟受限 | 三分法的第三支（另两支见 `memory-bound / compute-bound`）：算力有余、带宽也有余，**但并发度不足以把访存延迟盖住**。此时提高算术强度、提高带宽都不解决问题，该加的是**可并行的独立工作**。⚠ 它常被并进「带宽受限」——两者的处方相反（前者靠更多并发，后者靠更少字节）。 |
| headroom | 到上界的距离 | 「还能快多少」的可数形式：**上界 ÷ 当前**（`1.0` = 已贴上界，即「没有优化空间」）。**headroom gate**：上游实现里真实存在的一道闸——**一个 kernel 加速项声称的加速比不得超过该算子到其约束资源的距离**（边界**闭**：距上界 `1.300×` 时 `1.300×` 通过、`1.301×` 被拒）。⚠ 与 `speedup` 不同：headroom 是**比值上的天花板**，不是已实现的收益。 |
| tolerance（rtol / atol） | 容差 | 判定两份实现「一致」的界。**两种口径不可混**：`atol + rtol·|ref|` 与**绝对** `atol`——同一个 `1e-3` 在两种口径下**不是同一个界**。⚠ **容差不是 dtype 的函数**：同一算子、同一 dtype，输入动态范围从 1 档换到 3 档，实测所需 `rtol` 从 **`0`**（`atol` 就够）变成 **`3.7e-3`**，最大误差跨 **48.9 倍**；而现成的容差表通常比实测误差包络**松 1–3 个数量级**，把真错误吸收掉。 |
| determinism | 确定性 | 同一输入逐位可复现的性质。⚠ **它不是规范保证的**：PTX ISA 明写 `wmma.mma` 的**累加顺序未规定**；`atomicAdd` 系列（`index_add_` / `scatter_add_` / `bincount`）官方文档自陈 "no simple way of avoiding nondeterminism"。**所以「逐位可复现」是必须自己保证的性质**——契约里写 `bitwise_reproducible` 等于对实现提了一个规范不保证的要求。 |
| accumulation order | 累加顺序 | 浮点求和的次序。归约、`atomicAdd`、split-K 都会让它**不确定**；它同时决定误差随求和长度的增长形态。⚠ 它与 `determinism` 是**同一条链的两端**：顺序不定 ⇒ 结果不逐位可复现。 |
| run-to-run variance | 跑间波动（σ） | 同一实现重复测量的离散程度。**它是本域几乎所有判定的分母**——最小可检出差异 ≈ `4σ/√n`（两组等样本）。⚠ 只报一个数而不报 `σ` 与 `n`，那个数**无法用来判定**；⚠ **紧得可疑的 σ 反而可疑**——它可能来自**确定性输出**而非稳定性能。 |
| warmup（benchmark 预热） | 预热 / 丢弃前导测量 | 丢掉前 `k` 次测量以避开冷启动期。⚠ **正确做法是丢弃前 `k` 次，而不是靠中位数稳健**：冷启动值被保留时，中位数名次被推向稳态**上尾**，`k/n = 0.49` 时偏差已达 **`+0.1026`**（> 一个 σ），而 `k/n = 0.10` 只有 `+0.0072`。⚠ 官方 `do_bench` 的 `warmup=25` 单位是**毫秒**（不是 25 次）；不足时同一个 kernel 能「凭空快 27.8%」。★ **同名不同指**：本条一律指 **benchmark 预热**（丢弃前导测量）；训练里的**学习率 warmup** 是**另一件事**（见「训练与推理系统」节的 `学习率 warmup` 条）。 |
| steady state | 稳态 | 冷启动之后的性能分布；**测量值 = 稳态 + 冷启动摊销**。两条反直觉的定量事实：`k < n/2` 时中位数污染**被稳态上尾封顶在 ≈ `2σ`、与冷启动倍数无关**（1.5×–9× 同值），把 `σ` 放大 10 倍上界也放大 10 倍 ⇒ **决定污染大小的是「稳态有多散」，不是「冷启动有多慢」**。 |
| regression gate | 防回归门限（speed gate） | 判定「这次改动是不是真的更快」的门限。**必须逐架构标定，不能跨架构共用**（[C] 预印本与 [D] 社区项目两个独立主体都主张）。⚠ **同一改动跨架构可以反向**（L40S `+9.4%` / A100 SXM4 `−50%`，[C] 级预印本，措辞用「已有报告显示」）；⚠ 门限本身会被**测量噪声**越过——一次纯粹的 warmup 差异就给出 `+27.8%` 的假加速。 |

## 性能工程与测量

> **本节的词条正文在 `50-performance`。** 那一域的统一主线是
> **「工具的每一个输出都带口径。你看到的不是你的程序，是关于你的程序的一次测量」**——
> 下表每一条都是那条主线的一个**具体泄漏点**。

| 英文 | 中文 | 说明 |
|---|---|---|
| sampling (profiler) | 采样 | 周期性中断程序、抓**此刻栈上有什么符号**的采集机制。开销**与调用次数无关**，代价是**只能看见栈上有符号的东西**——解释器 / JIT / 内联会让函数边界消失。⚠ 它的失败形态是**归因失效**，不是报错：本机实测（`50-performance/profiling-tools/01`）被测程序里耗时最长的函数在报告中出现 **0 次**，榜首是内核计时调用 `mach_absolute_time`。⚠ 它与**插桩**是两条**结构性不同**的路，不是同一件事的两种实现。 |
| instrumentation (profiler) | 插桩 | 在每次函数进入/退出处埋 hook 的采集机制。换来**精确调用次数**与 **inclusive / exclusive 两套时间**，代价是**开销正比于调用次数**，且**改变被测程序的时序**。⚠ 「插桩开销」这个数**必须带上循环结构**才成立：本机实测同一段负载，**定次数**循环下 **+210.3%**、**wall-bounded** 循环下 **+0.0%**（另一次独立复跑为 **+174%~+276%** / **+0.1%**）。不带口径报这个数就是**量法口径错**。 |
| replay | 重放 | 硬件计数器**一次能同时读的数量有限**，所以 profiler 要把 kernel **重复执行多遍**来收齐指标。⚠ replay 之间要做 save/restore、**默认清 cache**、**默认锁时钟** ⇒ **每一遍的执行条件都不同**，而这些默认**不出现在报告里**。 |
| pass | 遍 | 收集指定一组指标的一次执行。⚠ 硬件计数器**一次能同时读的数量有限**，所以**一次 pass 收不齐所有 metric**；而**每一遍的执行条件都不同**（见 `replay`）。 |
| multi-pass | 多遍 | 一次收不齐就把 kernel 分多次执行来凑齐指标。⚠ 后果是**比值型指标的分子分母可能来自不同的 pass**：`hit rate = hits / queries` 的分子取自第 1 遍、分母取自第 2 遍时，那个比值**不是任何一次执行的测量结果**（本机对照：两次真实执行给出 28.5714% 与 0.0244%，四个真实可得的数两两不同，而混起来的两个值都不在其中；工具**故意不 clamp**，还能给出 400% 的命中率）。 |
| attribution | 归因 | 把一段耗时算到「谁头上」。⚠ 它是**会失效的**：采样器只能按**栈上存在的符号**归因，于是**你写的函数可以一次都不出现**，时间被记给机器栈上的 C 函数。失效**不报错**——报告看起来完全正常，只是榜首是别人的符号。这与 GPU 上「内联使函数边界消失」「JIT 代码没有符号」是**同一类问题**。 |
| inclusive vs exclusive | 含子调用 / 不含子调用 | **同一个函数名下的两个不同的量**：含子调用（`cumtime`）与不含（`tottime`）。⚠ 这是**同名不同指**的典型，不区分就会把它们当成一个。判据很直接：`cumtime − tottime` 应当**恰好等于**其子调的 exclusive 时间之和。报「热点」用 exclusive，报「这个子树的代价」用 inclusive，**两者不可互换**。 |
| 量化步长 `τ_q` | 计时器的单次量化步长 | **单次测量**的读数只能落在它的整数倍上 ⇒ **任何短于 `τ_q` 的差异，单次测量在原理上看不见**。本机 `perf_counter` 的底层是 `mach_absolute_time()`，`resolution = 41.6667 ns`（`= 125/3`，时基 **24 MHz**）。⚠ **三种「分辨率」口径不可混**：① 单次量化步长 `τ_q`；② **重复 N 次求平均的精度**（远好于 `τ_q`，但受固定开销支配）；③ `timeit.timeit("pass")` 报的 **5.48 ns**——那是**被测操作自身的成本**，比 `τ_q` 还小 **7.6 倍**，**不是计时器的精度**。⚠ 真实时间短于 `τ_q` 时读数会**恒为一个刻度** ⇒ 实测 `σ = 0` ⇒ `δ* = 0` ⇒ 判据退化（`45-kernel-lifecycle/convergence-and-tuning/02-significance`）。**不设 `τ_eff` 一词**：`τ_eff ≈ τ_q/√N` 这个关系是猜的，本机实测（N≥10 后 stdev 不随 N 单调变化）**不支持**。 |
| 重复次数的选择效应 | N 如何改变你报出的「单次耗时」 | ★ 本域实测：`t(N)/N` **不是 N 的单调函数，是 U 形**。小 N 端被**固定开销** `a`（循环建立 + 两次读数；本机 `N = 0` 的空循环 **83.0 ns**，拟合 `a = 74.8 ns`）抬高，大 N 端有一个**小幅回升**（高 N 段与平台段的中位比 **1.0163**，8 组复跑落在 **1.0067–1.0482**，**8/8 为正**）。⚠ 所以「单次耗时」**必须先说清 N**：同一个操作（一次约 275 ns）`N = 1` 报 **333.2 ns**、`N = 200`（谷底）报 **263.3 ns**，**差 26.5%**。⚠ **不存在「该重复多少次」的解析公式**——它是**一台机器一条曲线，必须自己标定**（两端都有系统性偏差，中间有一个谷底）。 |
| σ 的组间与组内成分 | within-run / run-to-run σ | 一批测量的方差可拆开：`σ²_total = σ²_run-to-run + σ²_within-run`。组内 = 「同一次运行内反复测」的抖动；组间 = 组间均值的标准差**扣掉组内噪声能解释的那部分**（`√(σ²_between_raw − σ²_within/m)`，**小于 0 时取 0**）。⚠ 它的用途是回答「**加运行次数，还是加单次内的重复数**」：**加 `m` 只压 `σ_within/√m`，`σ_between` 一点都不动**（后者只能靠换一次运行来采样）。⚠ 引用 σ 必须连**口径**一起引（哪个操作、哪个 N）：本机只改 N，同一操作的相对离散度差 **12.3 倍**。 |
| `δ*` | 最小可检出差异（MDE） | `δ* = (z_{α/2} + z_β) · σ · √(2/n)`，本库代入的系数是 **2.801585218112968**（80% 检定力、α = 0.05）。⚠ 因子是 **`√(2/n)`**（不是 `√(1/n)`），系数是 **2.80**（不是 4.0——那会让样本量多 **2.039 倍**）。判定：`\|Δ\| > δ*` 才算「看得见」。⚠ 它要求**两组口径可比**——两组的相对噪声不可比时 `δ*` **没有意义**；⚠ `σ` 本身不可复现（机器负载）时 `δ*` 也不可复现。⚠ `δ*` 是**能看见的最小量**，不是**重要的量**。公式与推导的正文在 `45-kernel-lifecycle/convergence-and-tuning/02-significance`，本域**只代入、不重推**。 |
| interleaving | 交错（A/B/A/B，且 ABBA） | 两个配置比较时的编排：**每一对里 A、B 各一次**，而不是「先跑完 A 再跑完 B」。⚠ 它针对的是**两个测量时间段之间的系统状态差异**——**不是**冷启动（预热后效应仍在）、**也不是**周期性干扰（构造的周期干扰造不出可观测效应；两条否定结论都实测过）。本机实测：A 与 B **逐字等价**（真实差异严格为 0）时，顺序编排给出 `\|差\|` 均值 **1.0262%**（8 轮里出现过 **+3.289%** 的「改进」），交错把它压到 **0.0989%**。⚠ 它**消不掉「负载影响」**，防的是**时间相关的漂移**；⚠ **永远先测同一组**会让它失效（本机 `\|差\|` 均值 0.02% → **0.186%**）⇒ 必须 **ABBA**；⚠ 干扰在时间上**均匀**分布时，交错**不带来优势**（没有可治的东西）。 |
| paired comparison | 配对 | 交错编排的**配套条件**：样本必须成对，**每一对里 A、B 各一次**。⚠ 两组样本数不等或未配对时，**交错的优势丧失**。⚠ 与「把两个断言配对使用」这类**普通用词**不是同一件事——`45-kernel-lifecycle` 全树 grep「配对」只有 3 处，全是后者。 |

## 编译器与图优化

| 英文 | 中文 | 说明 |
|---|---|---|
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
| `torch.compile(dynamic=…)` / `automatic_dynamic_shapes` | 动态形状开关 | ⚠ 方向必须写准：**`dynamic` 默认是 `None`（不是 `False`）**，且 **`automatic_dynamic_shapes` 默认 `True`**——先按**静态**形状编一次，guard 失败后**自动把变化的那些维标成 dynamic** 再编一次，此后新形状命中同一张动态图。**所以「N 个不同 shape 就重编译 N 次」是错的；重编译次数的期望是 O(变化的维度数)。** ⚠ **`dynamic=True` 不是「打开动态形状的开关」**：官方把它标成 *"(Not recommended)"*、*"error prone"*，归在 *"for testing only"*——那个需求**默认就已满足**；提前声明要用 `mark_dynamic` 逐维**用户标注**（官方推荐）。 |
| layout conversion | 布局转换 | 块级程序里某个值的**线程/寄存器分布**需要改变时，编译器插入的转换。⚠ 它**不是一个数**：有**免费**的转换（官方原话 `which is free`），也有要经共享内存**写一遍读一遍**（`2 × 元素数 × 字节宽`）加 barrier 的——同一行源码只差一个 `order` 就跨过这条线。⚠ 与 XLA 的 `layout assignment` **不是同一件事**：那是**选布局**的编译期阶段，这是**执行那次改布局**的代价。 |

## 深度学习框架

| 英文 | 中文 | 说明 |
|---|---|---|
| autograd | 自动微分 | 依据计算图自动求导的机制。**反向图是在前向时建的**（留什么见 saved tensors 条，挂点见 `grad_fn` 条）。⚠ 三条常被记反：**只有 `backward()` 写 `.grad`（且是累加），`torch.autograd.grad()` 只返回、不写**；**`retain_graph` 的默认值不是 `False`，而是 `create_graph`**（源码 `if retain_graph is None: retain_graph = create_graph`；因为 `create_graph` 默认 `False`，实践中才表现为 `False`）；**判叶子必须用形式判据**（见 `is_leaf` 条），口语的「用户创建的」在 view 上会给出相反答案。 |
| dispatch (dispatcher) | 分发 | 框架里「**按 key 分发的调度层**」。一次 `a + b` 恰好 **1 次** `__torch_dispatch__` 拦截，而 `+a`、已同 dtype 的 `a.float()`、`a.shape` 是 **0 次**——⚠ **「0 次」有两种成因，别混**：`a.shape` 是**根本没进 dispatcher**，`+a` / `a.float()` 是**进了但被内联展开**，判据是两条观测通道的**差集**。⚠ 数层数要按**实际算子**数，不能按脑子里的数学运算数：一维 `a @ b` 走的是 `aten.dot.default`，不是 mm/matmul。 |
| dispatch key / DispatchKeySet | 分发键 / 分发键集合 | keyset **是一个 64 位整数**，且**不是 1 key = 1 bit**（低位是 backend bits、高位是 functionality bits），backend 上限 **16**。⚠ **「张量子类就会有 `Python` key」是错的**——只有**覆盖了 `__torch_dispatch__`** 的子类才会被加上 `Python` 与 `PythonTLSSnapshot`，空子类不会。 |
| `__torch_dispatch__` / TorchDispatchMode | 算子级拦截 / 分发模式 | 两者拦的都是 **aten 层的算子**。`TorchDispatchMode` **不是 keyset 里的一位**，而是**线程局部栈**（叠加时后进先出），且进/出 mode 本身有成本（本机实测：进/出 6.4 μs，裸 `a + b` 0.9 μs）——**它是诊断工具，不是生产路径**。⚠ 两者同时在栈上时**顺序是 `['mode', 'subclass']`，即 mode 先**；这条的唯一依据是实测（源码里讲到顺序的那条注释只在 2.13.0 存在，且讲的是 C++ `Fake` key 的位置）。⚠ 在子类的 `__torch_dispatch__` 里直接 `func(*args)` 会**无限递归**（用 `as_subclass` 拆包**同样**递归），要 `with torch._C._DisableTorchDispatch():`。 |
| aten / CompositeImplicitAutograd | — | aten 是**算子层**（全名形如 `aten::add.Tensor`）。⚠ 注册为 **CompositeImplicitAutograd** 的算子（`reshape` / `ravel` / `contiguous` / `+a` / `to`）**对 dispatch mode 不可见**——它们被**内联展开**，拦到的是展开出来的子算子（`reshape` → `view`/`clone`）。**所以判据必须写成「那个算子自己是否出现在名单里」，不能写成「拦截数是否为 0」。** ⚠ `_dispatch_has_kernel` 要写重载名（`aten::add.Tensor`）**只对 aten 成立**：自定义算子用**裸名**，加 `.default` 反而抛错。 |
| `grad_fn` / `next_functions` | 反向节点 | **`grad_fn` 挂在输出张量上**，不在输入上。`next_functions` 是 `[(Node, int)]`，那个 `int` 是「这个 node 的第几个输入」；`None` = 该输入不需要梯度；**叶子输入对应 `AccumulateGrad`**——它才是把梯度累加进 `.grad` 的节点。 |
| `is_leaf` | 叶子张量 | 口语说法「用户创建的张量是叶子」**在 view 上会给出相反答案**（`w.view(-1)` 是用户创建的，但 `is_leaf = False`、`_is_view() = True`）。**形式判据是 `grad_fn is None or requires_grad is False`**，凡要对 view 判叶子必须用它。 |
| saved tensors | 保存的张量 | **autograd 这一层「固化」下来的东西**。留什么由**反向公式**决定、**逐算子不同**——本机实测同一组算例有 0 / 1 / 2 三种答案（有的存**输入**、有的存**输出**、标量乘一个都不存），⚠ **「一次前向保存 N 个中间结果」这类通用值是假的**。唯一通用观测手段是 `saved_tensors_hooks`；`grad_fn.saved_tensors` 在 **C++ 定义的节点上根本不存在**。 |
| `inference_mode` / `no_grad` | 推理模式 / 无梯度模式 | ⚠ 三条常被记反：① **出作用域后读是正常的**（`.shape`/`.sum()`/`.item()`/`.numpy()`/`.clone()`/`.detach()` 全通），只有**原地改**与**被 autograd 记录（save-for-backward）**被拒；② `_version` **在 mode 内就已不可读**（不是「出作用域才出问题」）；③ **forward-mode AD 在 2.11 与 2.13 之间行为相反**——老版本**静默给出全零 tangent、不抛异常**，一个「看起来能跑」的梯度是错的。 |
| view / reshape / contiguity | 视图 / 重排 / 连续性 | ⚠ **`view` 的判据不是 `is_contiguous()`**，而是 **contiguity-like condition**（`stride[i] = stride[i+1] × size[i+1]`）——**非连续**的 `randn(4,4)[:, ::2]` 照样能 `.view(8)`；`(4,4).t().view(16)` 失败是因为它**跨了两个连续段**。**`reshape` 在非连续时不一定拷贝**（先试 view，不行才拷）。`is_contiguous()` **只吃 sizes/strides/numel**：`numel == 0` 恒 True、**跳过 `size == 1` 的维**、既不查边界也不查重叠（签名里**没有 storage**）；`contiguous()` 在已连续时返回 **`self`** 且 **0 次 dispatcher 拦截**。⚠ 造反例不能用 `as_strided`（它**自带边界检查**），要用 `_make_wrapper_subclass`（它**必须**定义 `__torch_dispatch__`）。 |
| stride / storage_offset | 步长 / 存储偏移 | **形状不是布局，stride 才是**——`base.t()` 与 `base.reshape(6,4)` 形状相同、stride 不同。⚠ 判「是不是同一块存储」要用**未加 offset 的** `untyped_storage().data_ptr()` 配 `storage_offset()`：只看 `data_ptr()` 会把同一 storage 的两个不同区域判成两块，**两个判据给出相反结论**。 |
| tracing / retracing | 追踪 / 重追踪 | 追踪期的值是**符号值**（有形状与 dtype、**没有数据**），所以「拿它当布尔用」「迭代它」都会失败。**追踪会把每条路径都跑一遍**，因此 Python 副作用只在**第一次调用**时发生。⚠ **三者（TF / JAX / PyTorch）默认都对 shape 敏感**——没有哪一个默认不敏感；差别只在**缓解手段**。 |
| `TraceType` / `FunctionType` | — | TF `tf.function` 的缓存键：各实参的 `TraceType` 拼成的 `FunctionType`。`Tensor` 看 **dtype + shape**，`Variable` 还带一个唯一 resource ID，**Python 原生值看值本身**（`3` 是 `LiteralTraceType<3>`，**不是 `int`**）。匹配是**子类型**匹配、取最具体者：`[1,2] <: [None,None]` 成立、**反向不成立**——所以「先用 `None` 维 trace」才划算。⚠ 层级是 `PolymorphicFunction`（缓存层）→ `ConcreteFunction` →（v2.21.0 还夹着一层 `AtomicFunction`）→ `tf.Graph`；**`FunctionDef` 是 proto 的序列化形态，不是运行时对象**。 |
| AutoGraph | — | TF 把 Python 控制流转成图算子的机制，**转换开关只有一条：条件是 Tensor 才转**。是 → `tf.cond` / `tf.while_loop`（**两支都进图**，运行时选）；不是 → 普通 Python 条件（**图里只固化走的那一支**）。⚠ 最坏的一类是用**自由变量 / 全局变量**当条件：`TraceType` **只看输入参数**，改了它**不重追踪**、静默沿用旧图——**不报错，只是结果错了**。 |
| jaxpr / ClosedJaxpr | — | JAX 一次 trace 的产物，**explicitly typed / functional / first-order / ANF**。文法 `{ lambda Var* ; Var+. let Eqn* in [ Expr+ ] }` 里**分号前是 `constvars`（被闭包进图的常量）、分号后是 `invars`（每次调用都要喂进来的输入）**——「进不进 cache key」决定进哪一边（**默认情况下的 Python 标量在 `invars` 里**）。`jax.make_jaxpr()` 返回的是 **`ClosedJaxpr`**（jaxpr + 它闭包住的常量）。⚠ `first-order` 这一条正是「`vmap` / `grad` 必须叠在图**之上**」的答案。 |
| vmap / batcher | 向量化变换 / 批处理规则 | `vmap` **不展开循环**：机制是给每个 primitive 定义一条 **batching rule（batcher）**，把被映射的轴**推进 primitive 的语义里**。因为 **batcher 拿不到 batch 是几**，图里根本没有 B——官方逐字 *the program stays the same size no matter the batch*，而手写循环是 *a full copy of the body per batch element*。⚠ 没写 batcher 的 primitive 会抛 `Batching rule for '…' not implemented`；⚠ 上面两句**只在教程正文里，不在 `jax.vmap` 的 docstring 里**。 |
| `pmap` / `pjit` / `shard_map` | — | 多设备那条路的三代写法，换的一直是「**谁来把假设写下来**」。`pmap` **已被官方劝退**（docstring 首句 *Old way of doing parallel map*），且**现在就是用 `jit()` + `shard_map()` 实现的**；`pjit` **已并入 `jax.jit`**（落点很实：`jax.jit` 的返回类型就叫 `pjit.JitWrapped`）；`shard_map` 与 `jit` 的自动分片 **complementary to, and composable with**。可数的那笔账：同一个全局求和，自动分片由**分区器插 1 个 `psum`**、`shard_map` 由**用户写 1 个**——差别只在假设由谁写。 |
| `lax.cond` / `scan` / `while_loop` | 结构化控制流 | **值相关的 Python 控制流在 JAX 里直接失败**，替代是这一套。`lax.cond` 的**谓词可以是 traced 值**（分支选择在设备上、运行时发生——与 TF 的 `tf.cond` 同一设计，代价是**两支都进图**）；⚠ **`while_loop` 不是反向模式可微的**（反向要**静态有界**的迭代次数），要反传就用 `scan` / `fori_loop`。 |
| SavedModel | — | 图离开进程时的落地格式。⚠ **「SavedModel 已被废弃」是过度概括**：`tf.saved_model.save` 在 v2.21.0 源码里**没有任何 `@deprecated`**，而且 **Keras 3 的 `Model.export()` 默认格式就是它**（`format="tf_saved_model"`）；被取代的是「把它当作 Keras 模型的**保存 / 再训练**格式」这件事（那用 `.keras`）。⚠ **导出 = 图 + 签名一起落地**：载入后**不产生任何追踪**，但形状假设被冻住（要留余量就用 `None` 维）。 |
| `torch.library.custom_op` / `tf.custom_gradient` | 自定义算子入口 | PyTorch：`custom_op` 有三条硬约束——`mutates_args` 是 **keyword-only 且无默认值**、被包装函数**必须有类型标注**（schema 由它推导）、**返回的是 `CustomOpDef` 对象不是函数**（有 `register_kernel` / `register_fake`，**没有** `register_impl`）；自定义算子在 eager 上**与内置算子同层数**（1 次拦截）——注册是**交出契约**，不是绕开 dispatch。TF：官方次序是**先试着用现有 Python 算子组合，不行才写自定义 op**；`tf.custom_gradient` 是**给已有的组合换反向**，不是注册新 op。 |

## 模型架构

> **本节词条的正文在 `80-model-architectures`**（正文在彼，此处回链）。那一域的主线是
> **「架构是一组取舍：每一个选择都在别处付账——参数、字节、数值条件、位置容量、或者训练稳定性」**；
> 下表每一条都是那条主线上的一个**具体取舍**。★ 反方向（该域对本节的注记方向）见
> `80-model-architectures/README.md`：「（正文在本域，回链 `00-meta/glossary.md`）」。

| 英文 | 中文 | 说明 |
|---|---|---|
| 残差流 (residual stream) | 残差流 | 一个 `L` 层模型里**唯一贯穿全部层**的那个张量（宽度 `hidden_size`、长度 = 序列长）。一个 block 只做两件事：**读残差流、写一个修正量**。★ 归一化放在加法**内**还是加法**外**，决定的不是「哪个更好」，而是**残差流的范数随深度涨（pre-LN）还是不变（post-LN）**。（正文在 `80-model-architectures/transformer/01-block-and-residual-stream.md`，此处回链） |
| pre-norm / post-norm | 前置 / 后置归一化 | 两种接法：`h + F(LN(h))`（归一化在残差**里面**）与 `LN(h + F(h))`。★ 后果分两半：post-LN 每层末尾把范数钉在 **`√D`**（`‖h_l‖ = √D` 在 eps=0 时是**恒等式**）；pre-LN 的范数**随深度增长** ⇒ 末层**必须**补 `final_norm`。★ 引用纪律（原论文自己的措辞）：结论词是 **`comparable`**、不是「更好」；`outperform` 全文出现 **1** 次、**主语是 `Wang et al. 2019`**；post-LN 梯度那句原文用 **`hypothesize`**。（正文在 `80-model-architectures/normalization/02-placement-and-qk-norm.md`，此处回链） |
| `final_norm` | 末层归一化 | pre-LN 栈在 `lm_head` 之前补的那一次归一化。理由是一个**可手算的闭式**：`std(logits) ≈ ‖h_L‖ / √d` —— 不做它，logits 的尺度就与**深度耦合**（本机 24 层实测：未归一化 std `15.9234` → 归一化后 `1.0005`，比值 **15.915×**）。★ 判据：比值 `> 10×`。（正文在 `80-model-architectures/transformer/01-block-and-residual-stream.md` 的 §1.4 与判据③，此处回链） |
| SwiGLU 的 `8d/3` | — | ★★ **它是推导，不是论文里的话**：论文只给「把隐藏单元数**缩小 `2/3`**」（原始字节计数：`\frac{2}{3}` = **2**、`8d/3` = **0**、`4d` = **0**）。由 `3·d·d_ff^{GLU} = 2·d·d_ff^{两层}`，**在前提「朴素 FFN 取 `d_ff = 4d`」（论文之外的约定）下**才得到 `(2/3)·4d = 8d/3`。⚠ **真实模型都不用这个数**（Llama-3-8B 的 `intermediate_size / hidden_size = 3.5`、Qwen3-30B-A3B 是 `3.0`）⇒ 「与两层 FFN 参数持平」这句话**只在 `4d` 这条前提上成立**。（正文在 `80-model-architectures/transformer/02-ffn-shapes.md`，此处回链） |
| GQA / MQA | 分组/多查询注意力 | 多个 query 头共用一份 KV，**群组大小 `r = heads_q/heads_kv`**（文献惯例；MHA 时 `r=1`，MQA 是 `heads_kv=1` 即 `r=heads_q`）。decode 阶段**算术强度在数值上就等于 `r`**（fp16），**与序列长度和 head 维度都无关**——所以长上下文只让绝对时间线性变长，不改变瓶颈性质。（正文在 `80-model-architectures/attention-variants/01-mha-mqa-gqa.md`，此处回链） |
| MLA (Multi-head Latent Attention) | 多头潜注意力 | 动的是**另一个因子**：把 K 与 V **联合**低秩压到一个 `d_c` 维潜向量 `c`（`c = x·W_DKV`、`k = c·W_UK`、`v = c·W_UV`），**cache 里存的是 `c`**。★ 这个新结构带来一个原结构没有的问题：用的时候要不要把 `k` 算回来（「吸收」）——**没有位置编码时可以**（`W_eff = W_Q·W_UK^T` 与序列长无关，可离线合并一次），**有 RoPE 时不行**（旋转矩阵夹在 `q` 与 `c` 之间，而矩阵乘不交换）；代价是 KV 元素数为 **`(d_c + d_h^R)·l`**，**不是** `d_c·l`。（正文在 `80-model-architectures/attention-variants/02-mla-low-rank.md`，此处回链） |
| MoE (Mixture of Experts) | 混合专家 | 每 token 只过一个子集的专家（`top_k` 个）。计算量 `4·tokens·top_k·d_model·d_ff`（**两个 GEMM**：up + down；SwiGLU 三投影则是 6）——**与专家总数无关**。⚠ **省的是 FLOPs，不是显存**：全部专家的参数都要驻留，**权重流量随专家数线性上涨**，所以权重流量的算术强度 `2·T·top_k/(E·esize)` 随 `E` **反比下降**，拐点批大小 `T* = I*·E·esize/(2·top_k)` 随 `E` **线性上涨**。（定义正文在 `80-model-architectures/moe/01-gating-and-routing.md`，此处回链；⚠ 上面**算力 / 权重流量 / 拐点批大小**那一段的正文在 `10-foundations/performance-models`，**不在本域**） |
| 容量因子 (capacity factor) | — | 每个专家开出的 slot 数上限。**定义与语义的正文在 `80-model-architectures/moe/02-capacity-factor.md`**，kernel 侧只关心它带来的三种状态：正常、被截断（要 mask）、以及**`k` 个选择全被截断**——最后这种 token 前向输出恰为 0、反向梯度也恰为 0，**在本步训练里完全不存在**。 |
| 负载均衡损失 (aux loss) | 辅助损失 | Switch 式 (4) 逐字：`loss = α · N · Σ f_i · P_i` —— ★ **漏掉 `×N` 是最常见的错**（原文的理由逐字：让 loss 不随专家数变化）；`α` 是扫过 `1e-1`…`1e-5` 后取的 **`1e-2`**。`f_i` = 分给专家 `i` 的 token 占比；`P_i` = 路由概率在**全 `E` 个专家**上的列均值（**不是** top-k 归一化后的那 `k` 个权重）。★ 下界 `1.0` 是**代数的**（均匀时精确 `1`），不是有限样本的（实测 `1.0008` / `1.0012`）。⚠ GShard 的 aux loss 是**另一条式子**，不可与 Switch 的混用。（正文在 `80-model-architectures/moe/03-balance-and-granularity.md`，此处回链） |
| RoPE 的 `θ_i` / 混叠周期 | — | `θ_i = base^{-2(i-1)/d}`（`i = 1 … d/2`）是一条约**几何阶梯**（高频密、低频疏）。★ **`θ₁ = base⁰ = 1` 是精确的，且与 `base` 无关** ⇒ 第 1 对的**混叠周期精确等于 `2π`**：位置 `m` 与 `m + 2π` 在最高频分量上不可区分。⇒ **提高 `base` 只把低频端拉长**（`λ_{d/2}`），最高频那一维**一动不动**。★ 「能分辨到多长」写成 `turns_i(S) = θ_i·S/2π`（≡ YaRN 的 `r = L/λ`）；它两端是**两种不同的失败**（高端混叠、低端分辨不出），且**两者不可能同时满足**。（正文在 `80-model-architectures/positional-encoding/02-spectrum-and-aliasing.md`，此处回链） |
| 相位比 `Φ_ratio` | — | 本域写出的统一量（**[E]**）：`Φ_ratio_i := 扩展窗口下第 i 维的最大相位 ÷ 训练窗口下它的最大相位`；它的意义只有一句——**`Φ_ratio_i ≤ 1` 才叫「没有外推」**。⇒ 四种 RoPE 外推方法（PI / NTK-aware / NTK-by-parts / 不做任何事）是**同一个一参数族的四个成员**，差别只在**每维的频率缩放**：PI 恒 `1`（把不该缩的高频也缩了）；NTK-aware 的 `i = 1` 那一维 `= s` ⇒ **那一维被外推**；NTK-by-parts 恒落在 `[1/s, 1]`。（正文在 `80-model-architectures/positional-encoding/03-extrapolation.md`，此处回链） |
| ALiBi | — | **另一条路**：不加位置编码，直接在 logits 上加一项与 `(i−j)` 成正比的偏置 —— `softmax(q_i K^⊤ + m·[−(i−1), …, −1, 0])`，其中 `m` 是**训练前固定的、每个 head 一个的 slope**（闭式 `slope_h = (2^{−8/n})^h`）。★★ **脚注 10 逐字**：那条偏置**不乘** `1/√d_k` ⇒ 近因效应与 `d_k` **无关**。位置越新权重越大 ⇒ 它是一个**纯近因核**。（正文在 `80-model-architectures/positional-encoding/03-extrapolation.md`，此处回链） |

## 训练与推理系统

> ★ **本节与 `120-inference` 共享**：**训练部分**的正文在 `90-training`；
> **推理部分**的正文在 `120-inference`（未建成）。⇒ 下表每条都在自己的「说明」列里回链到正文所在处。
> （★ 另两条不属本节的归口：`ZeRO` / `FSDP` 的正文在 `100-parallelism` 的 `zero-fsdp`，此处只回链。）

| 英文 | 中文 | 说明 |
|---|---|---|
| KV cache | 键值缓存 | 自回归解码时缓存历史 token 的 K/V，避免重复计算。 |
| continuous batching | 连续批处理 | 在请求级别动态进出批次，而非等待整批完成。 |
| PagedAttention | 分页注意力 | 将 KV cache 按页管理，消除显存碎片。 |
| speculative decoding | 投机解码 | 用小模型草拟、大模型验证，减少解码步数。 |
| ZeRO | — | 将优化器状态、梯度、参数分片到各卡的显存优化方案。 |
| FSDP (Fully Sharded Data Parallel) | 全分片数据并行 | ZeRO 思想在 PyTorch 中的实现。 |
| all-reduce | 全规约 | 集合通信原语：所有参与者的数据求和后广播给所有人。 |
| flash attention | — | 用在线 softmax 分块，**不物化 `S×S` 矩阵**。收益不是「提高了算术强度」，而是**绝对字节数从 O(S²) 降到 O(S·d)**（S=16384 时 65 倍）。⚠ 朴素 attention 的 AI **渐近于 `d` 且随 S 上升**，不要照抄「S² 太大所以 memory-bound」。 |
| grouped GEMM | 分组矩阵乘 | 把变长的一批小 GEMM 合成一次调用，**N、K 相同、只有 M 变长**——所以问题归结为「M 不规整」。**按实际长度分组**的浪费只来自「`M_e` 不是 `BM` 的整数倍」（与分布无关）；**按最大专家填充**的浪费是 `1 − 均值/最大值`（均匀 ~4%、幂律 >60%）。⚠ 网格调度比填充策略重要得多：**一个专家一个 block** 只填满 `E/槽位数` 的机器，**一个 tile 一个 block** 能到 99%。 |
| reference implementation | 参考实现 | 与**被测实现独立**的对照实现。⚠ **独立性是硬要求，不是风格**：两者共享代码 `S` 时（`F = S∘f`、`R = S∘r`），比对 `S` 直接产出的量时差值**恒为 0**——捕获数 **`0`**、报出的最大误差 **`0.0`**，而它**看起来完全正常**（能通过最紧的容差）。机械判据两条：**参考实现不得调用被测实现的任何函数**（给共享代码装调用计数器，参考实现跑完后计数必须恰好为 `0`）、**且不得比被测实现更复杂**（否则出错时无法归因）。⚠ `maxdiff == 0.0` 有**两种相反含义**：同一实现的两条路径比 ⇒ 正确；共享代码的参考实现比 ⇒ **测不出来**。 |
| kernel cache key | kernel 缓存键 | 决定「这次调用复用哪个**已构建**的 kernel」的哈希键。**key 必须覆盖所有改变构建结果的字段**——dtype、layout、以及**可选参数是否出现**（是「是否」，不是它的取值）。⚠ 漏一项的后果是**静默**的：`key(A) == key(B)`、只构建 1 次、两个调用方拿到同一个对象，**没有任何一方看到错误**。⚠ 三类东西必须分开：**契约字段**（跨调用不变的**模式**，**不进** key）、**调用级输入**（**进** key）、以及 `stream` / device context 这类**运行期**值（两者都不是 ⇒ **不该**进 key）。 |
| contract（算子契约） | 算子契约 | 一个算子的接口声明。**判据不是「输入输出形状」，而是「是否改变构建出哪个 kernel」**——形状规则 / dtype / layout / 可选参数是否存在 / 输出精度 / 确定性 / 边界行为**都算**。⚠ **可测性判据**：一个字段**可测** ⟺ 它的取值域承诺了**可观察的区别**；被声明成 `unspecified` 的字段使这份契约**不存在任何能失败的测试**（可测性是 **`0`**，不是「较弱」）。⚠ 契约的作用域是**构建期**、后果出现在**测量期**——漏字段时不会有即时反馈，错误一路传导到性能数字上，那时已无法区分「设计不好」与「跑的其实是另一个 kernel」。 |
| initialization (He / Xavier) | 初始化 | 从哪开始：ReLU 系配 He（`Var[W] = 2/fan_in`）、tanh/linear 配 Xavier（`Var[W] = 2/(fan_in+fan_out)`，**均匀**分布）。★ Xavier 的 `2/(n_i+n_{i+1})` 是**前向与反向两个约束的折中**，**不是** `N(0, 1/n)`；★ He 只保**前向**（反向增益是 `2·fan_out/fan_in`）。⚠「只想同方差」时：`kaiming_uniform_(mode='fan_in')` 与 He **同方差**，`mode='fan_out'` 在 `(256,4096)` 上差 **16 倍**（= `fan_out/fan_in`）。（正文在 `90-training/initialization-and-stability/01-initialization.md`，此处回链） |
| master weights | 主权重 | fp32 的那一份权重副本 —— 优化器更新的是它，前向用的低精度副本每步从它降下来。★ **为什么必须宽**不是保守：判据是「**这个量跨不跨步存活**」，跨步存活的量把误差**攒起来**。★ 这条链上最容易出错的一步是**更新被静默吞掉**：`Δw < ULP(w)/2` 时 `w + Δw == w` **逐位**成立。（正文在 `90-training/mixed-precision/01-type-selection.md`，此处回链；★「必须 fp32」的**数值**理由在 `10-foundations/numerical-linear-algebra/04-iterative-methods.md:229-266`，正文在彼） |
| loss scaling | 损失缩放 | 反向前把 loss 乘 `S`、更新前除回来，用来把 fp16 下溢的梯度抬进可表示范围。★ `×S ÷S` 能是**恒等**这件事，**只因为 `S` 取 2 的幂**（换成 `1000` 就不再逐位相等）；★ 初值有**两个官方数**（NVIDIA 的 `2^24` 与 PyTorch 的 `65536`）。（正文在 `90-training/mixed-precision/02-amp-execution.md`，此处回链；★「为什么 fp16 要它、bf16 不要」**整块**的正文在 `10-foundations/floating-point/04-low-precision-formats.md`） |
| AMP / GradScaler | 自动混合精度 / 梯度缩放器 | 「哪些算子走低精度」由框架的 `autocast` 决定；`GradScaler` 负责 `S` 的自适应（成长 `×2`、回退 `×0.5`）与溢出检测（检测到就**跳步**，权重精确不变）。★ 最容易写错的是**一条顺序**：`unscale` 在 `backward` **之后**、`clip` **之前**。⚠ `autocast` 的算子白名单 / 黑名单**属框架 API 语义**（各框架互不相同），正文在 `70-frameworks/`，此处回链。（正文在 `90-training/mixed-precision/02-amp-execution.md`，此处回链） |
| 优化器状态 (`m`, `v`) | — | Adam 跨步存活的**一阶矩 `m`** 与**二阶矩 `v`**（以及 `step`），每步更新。★ 它**不在「参数量 `Ψ`」那一格**：类型选择与字节账都要单独算，并且它是「跨步存活 ⇒ 必须宽」的第二个实例（与主权重并列）。（更新式的正文在 `90-training/optimizers/02-adam-and-adamw.md`；字节账的正文在 `90-training/checkpointing/01-state-and-bytes.md`；此处回链） |
| gradient clipping | 梯度裁剪 | 按全局范数上限 `max_norm` 缩放梯度（`clip_grad_norm_` 那一族）。★ **阈值与判据的正文**在 `90-training/autograd-and-backward/03-gradient-lifecycle.md` 的表 A ⑦（在那里只写一次）；★ **监控量**的口径（取裁剪前还是裁剪后、判 `>` 还是 `isfinite`、全局还是逐张量）在 `90-training/initialization-and-stability/02-clipping-and-monitoring.md`。⚠ 它与「最优 clipping / ACIQ」那条的 `clip`（**量化**意义：学一个裁剪上界）**不是一回事**。 |
| 学习率 warmup（LR warm-up） | 学习率预热 | 训练开头**若干步**把 `lr` 从 ~0 抬到峰值，再交给调度衰减。★ **与「性能工程与测量」节的 `warmup（benchmark 预热）` 是同名不同指**（那条是丢弃前导测量）。★ 它**不改变总步长预算、只改变分配**：实测 `Σ lr(t) = lr_max·T/2`，**与 `W` 无关**。（正文在 `90-training/optimizers/03-schedules-and-clipping.md`，此处回链；★ **机制解释**（post-LN 的平均场理由）的正文在 `80-model-architectures/normalization/02-placement-and-qk-norm.md:83-100`） |
| loss spike | 损失尖峰 | 训练中 loss 在某一两步突然放大。★ **本库的立场**：机制层**没有一手结论** —— 能写的只有**现象**（Adam 的更新量在 spike 那一步**变小**、随后几百步被压低）、**已被证伪**的归因（坏数据；以及「开了 clip 就不会 spike」）、和**仍是假设**的归因（梯度范数突增）。（正文在 `90-training/initialization-and-stability/03-instability-and-spikes.md`，此处回链） |
| 重算换显存 (gradient checkpointing / recompute) | 重算换显存 | 前向**丢掉**中间激活、反向时**重算**它们，把 `n` 层的激活账压到 `O(√n)`；代价记在**时间**上（前向次数从 `n` 到 `2n − ⌈√n⌉`）。★ **两个口径必须分开**：`O(√n)` 是**内存**、`≈2n` 是**前向次数**。★★ **中文「检查点」不得用于它**；英文 `checkpointing` 在这里只在**论文术语**（Chen 2016）的意义上用。（正文在 `90-training/autograd-and-backward/02-gradient-checkpointing.md`，此处回链；★ 与本节的 `sharded checkpoint` 是**同名不同指**） |
| sharded checkpoint | 分片检查点 | 把训练状态切成 `N` 份分别落盘。★ **它不省总字节**（多出 `N × 每片头开销`），省的是**单卡峰值**；★ 而「拼得回来」**完全依赖元数据**（头里的 `start` / 摘要 / 覆盖率或全局元数据，**三层缺一不可**）。★ 检查点**不含梯度** ⇒ 它的字节账与「训练显存」（ZeRO 那个 `16Ψ`）**差一个梯度项**。（正文在 `90-training/checkpointing/02-sharded-save-and-load.md` 与 `90-training/checkpointing/01-state-and-bytes.md`，此处回链） |
